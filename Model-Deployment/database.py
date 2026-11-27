"""Persistence for violation instances, snapshots and alerts.

Two backends behind one class:

- SQLite (default): a file next to the app. Local runs and the test suite.
- PostgreSQL (`url` given): Azure Database for PostgreSQL in the cloud deployment,
  reached through a small connection pool. With a `token_provider`, every new
  connection signs in with a fresh Microsoft Entra token for the app's managed
  identity, so no database password exists anywhere.

Queries are written once with '?' placeholders; the cursor wrapper rewrites
them for psycopg. Rows are read by column name, never by position: `track_id`
and `owner` were added by migration after the first SQLite schema shipped, so
positional indexes silently point at the wrong column on upgraded databases.

Snapshot images are not stored here -- only their names. The bytes live in a
snapshot store (storage.py); this class asks it to delete them alongside rows.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from storage import LocalSnapshotStore

DATABASE = 'detections.db'

# Whitelist for the review page's ?sort= / ?order= parameters. They are
# interpolated into ORDER BY, which cannot take a bound parameter.
SORTABLE_COLUMNS = ('first_detected', 'last_updated', 'instance_id')
SORT_ORDERS = ('asc', 'desc')

SQLITE_SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id TEXT UNIQUE,
        first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_compliant BOOLEAN DEFAULT 0,
        missing_ppe TEXT,
        detected_ppe TEXT
    )''',
    '''CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id TEXT,
        snapshot_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (instance_id) REFERENCES instances(instance_id)
    )''',
    '''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        alert_type TEXT,
        description TEXT,
        snapshot_path TEXT
    )''',
)

# PostgreSQL starts at the current schema; there is no older one to migrate.
POSTGRES_SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS instances (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        instance_id TEXT UNIQUE NOT NULL,
        first_detected TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
        is_compliant BOOLEAN NOT NULL DEFAULT FALSE,
        missing_ppe TEXT,
        detected_ppe TEXT,
        track_id INTEGER,
        owner TEXT
    )''',
    '''CREATE TABLE IF NOT EXISTS snapshots (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
        snapshot_path TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
    )''',
    '''CREATE TABLE IF NOT EXISTS alerts (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
        alert_type TEXT,
        description TEXT,
        snapshot_path TEXT,
        owner TEXT
    )''',
    'CREATE INDEX IF NOT EXISTS instances_owner_idx ON instances (owner)',
    'CREATE INDEX IF NOT EXISTS instances_last_updated_idx ON instances (last_updated)',
    'CREATE INDEX IF NOT EXISTS snapshots_instance_idx ON snapshots (instance_id)',
    'CREATE INDEX IF NOT EXISTS snapshots_path_idx ON snapshots (snapshot_path)',
)


def _iso_utc(value):
    """A stored timestamp -> ISO 8601 UTC with 'Z'.

    SQLite returns 'YYYY-MM-DD HH:MM:SS' strings (UTC, no zone marker) and
    PostgreSQL returns aware datetimes. Without the marker the browser parses
    the string as local time, so a server in UTC shows every violation hours off.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    if not value or 'T' in value:
        return value
    return value.replace(' ', 'T') + 'Z'


def _split(value):
    return value.split(',') if value else []


def stored_snapshot_path(filename):
    """Canonical form of a snapshot path in the database: 'snapshots/<file>'.

    Independent of where snapshots are physically kept (a local directory, a
    container volume, a blob container), so moving storage does not orphan rows.
    """
    return 'snapshots/' + os.path.basename(filename)


class _Cursor:
    """DB-API cursor that accepts '?' placeholders on either backend."""

    def __init__(self, cursor, placeholder):
        self._cursor = cursor
        self._placeholder = placeholder

    def execute(self, sql, params=()):
        if self._placeholder != '?':
            sql = sql.replace('?', self._placeholder)
        self._cursor.execute(sql, tuple(params))
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount


def _postgres_pool(url, token_provider):
    """Connection pool for PostgreSQL; signs each new connection in with a token."""
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    class TokenConnection(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo='', **kwargs):
            if token_provider is not None:
                # Entra access tokens expire, and PostgreSQL checks the token only
                # at sign-in. Fetching one per new connection (the credential
                # caches and refreshes it) and recycling connections every 30
                # minutes means no connection ever signs in with a stale token.
                kwargs['password'] = token_provider()
            return super().connect(conninfo, **kwargs)

    pool = ConnectionPool(
        url,
        connection_class=TokenConnection,
        kwargs={'row_factory': dict_row},
        min_size=1,
        max_size=5,
        max_lifetime=30 * 60,
        open=False,
        name='safesite',
    )
    pool.open(wait=True, timeout=60)
    return pool


class Database:
    """Database handler for detections and alerts."""

    def __init__(self, db_path=DATABASE, snapshot_dir='snapshots', url=None,
                 token_provider=None, store=None):
        self.db_path = db_path
        self.store = store if store is not None else LocalSnapshotStore(snapshot_dir)
        self.dialect = 'postgres' if url else 'sqlite'
        self._pool = _postgres_pool(url, token_provider) if url else None

    def close(self):
        if self._pool is not None:
            self._pool.close()

    @contextmanager
    def _cursor(self):
        """One transaction: commits when the block succeeds, rolls back otherwise."""
        if self._pool is not None:
            with self._pool.connection() as conn, conn.cursor() as cur:
                yield _Cursor(cur, '%s')
            return

        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield _Cursor(conn.cursor(), '?')
        finally:
            conn.close()

    def _timestamp(self, moment):
        """A UTC datetime in the form this backend compares timestamps in."""
        if self.dialect == 'postgres':
            return moment
        return moment.strftime('%Y-%m-%d %H:%M:%S')

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self):
        """Create tables (and, on SQLite, apply migrations)."""
        try:
            with self._cursor() as c:
                if self.dialect == 'sqlite':
                    # WAL lets the review page read while the stream is writing.
                    c.execute('PRAGMA journal_mode=WAL')
                    for statement in SQLITE_SCHEMA:
                        c.execute(statement)
                    self._migrate(c)
                else:
                    for statement in POSTGRES_SCHEMA:
                        c.execute(statement)
        except Exception as e:  # noqa: BLE001 - report any backend's error the same way
            print(f"Error initializing database: {e}")
            raise

    def _migrate(self, cursor):
        """Add columns introduced after the first SQLite schema shipped.

        - instances.track_id: lets the review page separate two workers logged
          in the same minute.
        - instances.owner / alerts.owner: the visitor a row belongs to on the
          public demo, so one visitor's webcam snapshots are never shown to
          another.
        """
        def columns(table):
            cursor.execute(f'PRAGMA table_info({table})')
            return {row['name'] for row in cursor.fetchall()}

        instance_columns = columns('instances')
        if 'track_id' not in instance_columns:
            cursor.execute('ALTER TABLE instances ADD COLUMN track_id INTEGER')
            print('[Database] Migrated: instances.track_id added')
        if 'owner' not in instance_columns:
            cursor.execute('ALTER TABLE instances ADD COLUMN owner TEXT')
            print('[Database] Migrated: instances.owner added')

        if 'owner' not in columns('alerts'):
            cursor.execute('ALTER TABLE alerts ADD COLUMN owner TEXT')
            print('[Database] Migrated: alerts.owner added')

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def log_instance_snapshot(self, instance_id, missing_ppe, detected_ppe, snapshot_path,
                              track_id=None, owner=None):
        """Record a violation snapshot, creating the instance row on first sight."""
        if not instance_id or not snapshot_path:
            return False

        try:
            with self._cursor() as c:
                c.execute('SELECT id FROM instances WHERE instance_id = ?', (instance_id,))
                if not c.fetchone():
                    c.execute('''
                        INSERT INTO instances
                            (instance_id, is_compliant, missing_ppe, detected_ppe, track_id, owner)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (instance_id, False, ','.join(missing_ppe), ','.join(detected_ppe),
                          track_id, owner))
                    print(f"Created new instance record: {instance_id}")
                else:
                    c.execute('''
                        UPDATE instances
                        SET last_updated = CURRENT_TIMESTAMP, missing_ppe = ?, detected_ppe = ?
                        WHERE instance_id = ?
                    ''', (','.join(missing_ppe), ','.join(detected_ppe), instance_id))

                c.execute('INSERT INTO snapshots (instance_id, snapshot_path) VALUES (?, ?)',
                          (instance_id, snapshot_path))
            print(f"Logged snapshot for {instance_id}: {snapshot_path}")
            return True
        except Exception as e:  # noqa: BLE001 - logging must never stop the stream
            print(f"Error logging instance snapshot: {e}")
            return False

    def log_alert(self, alert_type, description, snapshot_path, owner=None):
        """Log an alert."""
        try:
            with self._cursor() as c:
                c.execute('''
                    INSERT INTO alerts (alert_type, description, snapshot_path, owner)
                    VALUES (?, ?, ?, ?)
                ''', (alert_type, description, snapshot_path, owner))
        except Exception as e:  # noqa: BLE001
            print(f"Error logging alert: {e}")

    def delete_instance(self, instance_id, owner=None):
        """Delete an instance, its snapshot rows and the images.

        With `owner` set, only that visitor's instance can be deleted.
        Returns False when nothing matched.
        """
        try:
            with self._cursor() as c:
                query = 'SELECT 1 AS found FROM instances WHERE instance_id = ?'
                params = [instance_id]
                if owner is not None:
                    query += ' AND owner = ?'
                    params.append(owner)
                c.execute(query, params)
                if not c.fetchone():
                    return False

                c.execute('SELECT snapshot_path FROM snapshots WHERE instance_id = ?', (instance_id,))
                paths = [row['snapshot_path'] for row in c.fetchall()]

                c.execute('DELETE FROM snapshots WHERE instance_id = ?', (instance_id,))
                c.execute('DELETE FROM instances WHERE instance_id = ?', (instance_id,))
        except Exception as e:  # noqa: BLE001
            print(f"Error deleting instance: {e}")
            return False

        # Rows first, then images: a crash in between leaves an orphaned image
        # (cleaned by retention), never a row pointing at a missing image.
        self._remove_snapshots(paths)
        return True

    def purge_older_than(self, hours):
        """Delete instances (and their images) untouched for `hours`, plus old alerts.

        Retention for the public demo: visitors' webcam snapshots should not
        outlive the visit. Returns the number of instances removed.
        """
        cutoff = self._timestamp(datetime.now(timezone.utc) - timedelta(hours=float(hours)))
        try:
            with self._cursor() as c:
                c.execute('''
                    SELECT s.snapshot_path FROM snapshots s
                    JOIN instances i ON i.instance_id = s.instance_id
                    WHERE i.last_updated < ?
                ''', (cutoff,))
                paths = [row['snapshot_path'] for row in c.fetchall()]

                c.execute('''
                    DELETE FROM snapshots WHERE instance_id IN (
                        SELECT instance_id FROM instances WHERE last_updated < ?)
                ''', (cutoff,))
                c.execute('DELETE FROM instances WHERE last_updated < ?', (cutoff,))
                removed = c.rowcount
                c.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff,))
        except Exception as e:  # noqa: BLE001
            print(f"Error purging old instances: {e}")
            return 0

        self._remove_snapshots(paths)
        return removed

    def _remove_snapshots(self, stored_paths):
        for stored_path in stored_paths:
            if stored_path:
                self.store.delete(os.path.basename(stored_path))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_statistics(self, owner=None):
        """Counts for the dashboard, optionally limited to one visitor."""
        try:
            with self._cursor() as c:
                owner_clause = ' AND owner = ?' if owner is not None else ''
                owner_params = (owner,) if owner is not None else ()

                c.execute('SELECT COUNT(*) AS n FROM instances WHERE is_compliant = ?' + owner_clause,
                          (False,) + owner_params)
                non_compliant = c.fetchone()['n']

                c.execute('SELECT COUNT(*) AS n FROM alerts WHERE 1 = 1' + owner_clause, owner_params)
                total_alerts = c.fetchone()['n']

            return {
                'total_detections': non_compliant,
                'non_compliant_count': non_compliant,
                'total_alerts': total_alerts,
            }
        except Exception as e:  # noqa: BLE001
            print(f"Error getting statistics: {e}")
            return {'total_detections': 0, 'non_compliant_count': 0, 'total_alerts': 0}

    def get_all_instances(self, sort_by='first_detected', sort_order='desc', owner=None):
        """Non-compliant instances with their snapshot counts.

        Raises ValueError for a sort column or order outside the whitelist.
        """
        if sort_by not in SORTABLE_COLUMNS:
            raise ValueError(f"sort must be one of {', '.join(SORTABLE_COLUMNS)}")
        sort_order = sort_order.lower()
        if sort_order not in SORT_ORDERS:
            raise ValueError("order must be 'asc' or 'desc'")

        # GROUP BY the primary key: PostgreSQL then accepts the other columns
        # of i as functionally dependent (a UNIQUE column would not do).
        query = '''
            SELECT i.instance_id, i.first_detected, i.last_updated, i.is_compliant,
                   i.missing_ppe, i.detected_ppe, i.track_id, i.id,
                   COUNT(s.id) AS snapshot_count
            FROM instances i
            LEFT JOIN snapshots s ON i.instance_id = s.instance_id
            WHERE i.is_compliant = ?
        '''
        params = [False]
        if owner is not None:
            query += ' AND i.owner = ?'
            params.append(owner)
        query += f' GROUP BY i.id ORDER BY i.{sort_by} {sort_order}'

        try:
            with self._cursor() as c:
                rows = c.execute(query, params).fetchall()
        except Exception as e:  # noqa: BLE001
            print(f"Error getting instances: {e}")
            return []

        return [{
            'id': row['id'],
            'instance_id': row['instance_id'],
            'first_detected': _iso_utc(row['first_detected']),
            'last_updated': _iso_utc(row['last_updated']),
            'is_compliant': bool(row['is_compliant']),
            'missing_ppe': _split(row['missing_ppe']),
            'detected_ppe': _split(row['detected_ppe']),
            'track_id': row['track_id'],
            'snapshot_count': row['snapshot_count'],
        } for row in rows]

    def get_instance_snapshots(self, instance_id, owner=None):
        """An instance and its snapshots, or None if absent (or not this owner's)."""
        try:
            with self._cursor() as c:
                query = '''SELECT instance_id, first_detected, last_updated, missing_ppe,
                                  detected_ppe, track_id
                           FROM instances WHERE instance_id = ?'''
                params = [instance_id]
                if owner is not None:
                    query += ' AND owner = ?'
                    params.append(owner)
                c.execute(query, params)
                instance = c.fetchone()
                if not instance:
                    return None

                c.execute('''
                    SELECT snapshot_path, timestamp FROM snapshots
                    WHERE instance_id = ? ORDER BY timestamp ASC
                ''', (instance_id,))
                snapshots = c.fetchall()
        except Exception as e:  # noqa: BLE001
            print(f"Error getting instance snapshots: {e}")
            return None

        return {
            'instance_id': instance['instance_id'],
            'first_detected': _iso_utc(instance['first_detected']),
            'last_updated': _iso_utc(instance['last_updated']),
            'missing_ppe': _split(instance['missing_ppe']),
            'detected_ppe': _split(instance['detected_ppe']),
            'track_id': instance['track_id'],
            'snapshots': [{
                'path': row['snapshot_path'],
                'url': '/snapshots/' + os.path.basename(row['snapshot_path']),
                'timestamp': _iso_utc(row['timestamp']),
            } for row in snapshots],
        }

    def snapshot_owner(self, filename):
        """(exists, owner) for a snapshot file name, used to gate image access.

        Paths are stored as 'snapshots/<file>' (see `stored_snapshot_path`);
        matching is exact, never LIKE, since '_' in a file name is a wildcard.
        """
        try:
            with self._cursor() as c:
                row = c.execute('''
                    SELECT i.owner FROM snapshots s
                    JOIN instances i ON i.instance_id = s.instance_id
                    WHERE s.snapshot_path IN (?, ?)
                    LIMIT 1
                ''', (stored_snapshot_path(filename), filename)).fetchone()
        except Exception as e:  # noqa: BLE001
            print(f"Error looking up snapshot owner: {e}")
            return False, None
        return (row is not None), (row['owner'] if row else None)

    def max_serial(self, date_prefix):
        """Highest serial already used for a day's instance IDs ('MM_DD_YYYY').

        A container restart loses the serial counter file; with a persistent
        database the counter must resume above what is already stored, or a
        new violation would reuse -- and overwrite -- an existing instance ID.
        """
        prefix = f"{date_prefix}_"
        try:
            with self._cursor() as c:
                # Range scan on the prefix instead of LIKE: '_' is a wildcard there.
                rows = c.execute('SELECT instance_id FROM instances WHERE instance_id >= ? AND instance_id < ?',
                                 (prefix, prefix + '~')).fetchall()
        except Exception as e:  # noqa: BLE001
            print(f"Error reading instance serials: {e}")
            return 0

        serials = [int(r['instance_id'][len(prefix):]) for r in rows
                   if r['instance_id'][len(prefix):].isdigit()]
        return max(serials, default=0)

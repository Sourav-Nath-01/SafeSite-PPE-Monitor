"""SQLite persistence for violation instances, snapshots and alerts.

Rows are read by column name (sqlite3.Row), never by position: `track_id` and
`owner` were added by migration after the first schema shipped, so positional
indexes silently point at the wrong column on upgraded databases.
"""

import os
import sqlite3
from contextlib import closing

DATABASE = 'detections.db'

# Whitelist for the review page's ?sort= / ?order= parameters. They are
# interpolated into ORDER BY, which cannot take a bound parameter.
SORTABLE_COLUMNS = ('first_detected', 'last_updated', 'instance_id')
SORT_ORDERS = ('asc', 'desc')


def _iso_utc(timestamp):
    """SQLite CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS', UTC) -> ISO 8601 with 'Z'.

    Without the zone marker the browser parses it as local time, so a server
    running in UTC (any cloud host) shows every violation hours off.
    """
    if not timestamp or 'T' in timestamp:
        return timestamp
    return timestamp.replace(' ', 'T') + 'Z'


def _split(value):
    return value.split(',') if value else []


def stored_snapshot_path(filename):
    """Canonical form of a snapshot path in the database: 'snapshots/<file>'.

    Independent of where the data directory is mounted, so moving the service
    (e.g. into a container with PPE_DATA_DIR=/data) does not orphan old rows.
    """
    return 'snapshots/' + os.path.basename(filename)


class Database:
    """Database handler for detections and alerts."""

    def __init__(self, db_path=DATABASE, snapshot_dir='snapshots'):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def snapshot_file(self, stored_path):
        """On-disk location of a snapshot, whatever form its path was stored in.

        Only the basename is trusted, so a stored or requested path can never
        point outside the snapshot directory.
        """
        return os.path.join(self.snapshot_dir, os.path.basename(stored_path))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self):
        """Create tables and apply migrations."""
        try:
            with closing(self._connect()) as conn, conn:
                c = conn.cursor()
                # WAL lets the review page read while the stream is writing.
                c.execute('PRAGMA journal_mode=WAL')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS instances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instance_id TEXT UNIQUE,
                        first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_compliant BOOLEAN DEFAULT 0,
                        missing_ppe TEXT,
                        detected_ppe TEXT
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instance_id TEXT,
                        snapshot_path TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (instance_id) REFERENCES instances(instance_id)
                    )
                ''')

                c.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT,
                        description TEXT,
                        snapshot_path TEXT
                    )
                ''')

                self._migrate(c)
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")

    def _migrate(self, cursor):
        """Add columns introduced after the first schema shipped.

        - instances.track_id: lets the review page separate two workers logged
          in the same minute.
        - instances.owner / alerts.owner: the visitor a row belongs to on the
          public demo, so one visitor's webcam snapshots are never shown to
          another.
        """
        def columns(table):
            cursor.execute(f'PRAGMA table_info({table})')
            return {row[1] for row in cursor.fetchall()}

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
            with closing(self._connect()) as conn, conn:
                c = conn.cursor()
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
        except sqlite3.Error as e:
            print(f"Error logging instance snapshot: {e}")
            return False

    def log_alert(self, alert_type, description, snapshot_path, owner=None):
        """Log an alert."""
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute('''
                    INSERT INTO alerts (alert_type, description, snapshot_path, owner)
                    VALUES (?, ?, ?, ?)
                ''', (alert_type, description, snapshot_path, owner))
        except sqlite3.Error as e:
            print(f"Error logging alert: {e}")

    def delete_instance(self, instance_id, owner=None):
        """Delete an instance, its snapshot rows and the image files.

        With `owner` set, only that visitor's instance can be deleted.
        Returns False when nothing matched.
        """
        try:
            with closing(self._connect()) as conn, conn:
                c = conn.cursor()
                query = 'SELECT 1 FROM instances WHERE instance_id = ?'
                params = [instance_id]
                if owner is not None:
                    query += ' AND owner = ?'
                    params.append(owner)
                c.execute(query, params)
                if not c.fetchone():
                    return False

                c.execute('SELECT snapshot_path FROM snapshots WHERE instance_id = ?', (instance_id,))
                self._remove_files(row['snapshot_path'] for row in c.fetchall())

                c.execute('DELETE FROM snapshots WHERE instance_id = ?', (instance_id,))
                c.execute('DELETE FROM instances WHERE instance_id = ?', (instance_id,))
            return True
        except sqlite3.Error as e:
            print(f"Error deleting instance: {e}")
            return False

    def purge_older_than(self, hours):
        """Delete instances (and their images) untouched for `hours`, plus old alerts.

        Retention for the public demo: visitors' webcam snapshots should not
        outlive the visit. Returns the number of instances removed.
        """
        cutoff = f'-{float(hours)} hours'
        try:
            with closing(self._connect()) as conn, conn:
                c = conn.cursor()
                c.execute('''
                    SELECT s.snapshot_path FROM snapshots s
                    JOIN instances i ON i.instance_id = s.instance_id
                    WHERE i.last_updated < datetime('now', ?)
                ''', (cutoff,))
                self._remove_files(row['snapshot_path'] for row in c.fetchall())

                c.execute('''
                    DELETE FROM snapshots WHERE instance_id IN (
                        SELECT instance_id FROM instances WHERE last_updated < datetime('now', ?))
                ''', (cutoff,))
                c.execute("DELETE FROM instances WHERE last_updated < datetime('now', ?)", (cutoff,))
                removed = c.rowcount
                c.execute("DELETE FROM alerts WHERE timestamp < datetime('now', ?)", (cutoff,))
            return removed
        except sqlite3.Error as e:
            print(f"Error purging old instances: {e}")
            return 0

    def _remove_files(self, stored_paths):
        for stored_path in stored_paths:
            if not stored_path:
                continue
            path = self.snapshot_file(stored_path)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                print(f"Error removing snapshot {path}: {e}")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_statistics(self, owner=None):
        """Counts for the dashboard, optionally limited to one visitor."""
        try:
            with closing(self._connect()) as conn:
                c = conn.cursor()
                owner_clause = ' AND owner = ?' if owner is not None else ''
                owner_params = (owner,) if owner is not None else ()

                c.execute('SELECT COUNT(*) FROM instances WHERE is_compliant = 0' + owner_clause,
                          owner_params)
                non_compliant = c.fetchone()[0]

                c.execute('SELECT COUNT(*) FROM alerts WHERE 1 = 1' + owner_clause, owner_params)
                total_alerts = c.fetchone()[0]

            return {
                'total_detections': non_compliant,
                'non_compliant_count': non_compliant,
                'total_alerts': total_alerts,
            }
        except sqlite3.Error as e:
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

        query = '''
            SELECT i.instance_id, i.first_detected, i.last_updated, i.is_compliant,
                   i.missing_ppe, i.detected_ppe, i.track_id, i.id,
                   COUNT(s.id) AS snapshot_count
            FROM instances i
            LEFT JOIN snapshots s ON i.instance_id = s.instance_id
            WHERE i.is_compliant = 0
        '''
        params = []
        if owner is not None:
            query += ' AND i.owner = ?'
            params.append(owner)
        query += f' GROUP BY i.instance_id ORDER BY i.{sort_by} {sort_order}'

        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as e:
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
            with closing(self._connect()) as conn:
                c = conn.cursor()
                query = 'SELECT * FROM instances WHERE instance_id = ?'
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
        except sqlite3.Error as e:
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
            with closing(self._connect()) as conn:
                row = conn.execute('''
                    SELECT i.owner FROM snapshots s
                    JOIN instances i ON i.instance_id = s.instance_id
                    WHERE s.snapshot_path IN (?, ?)
                    LIMIT 1
                ''', (stored_snapshot_path(filename), filename)).fetchone()
        except sqlite3.Error as e:
            print(f"Error looking up snapshot owner: {e}")
            return False, None
        return (row is not None), (row['owner'] if row else None)

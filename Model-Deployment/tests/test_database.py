"""Tests for the violation log: schema, query safety, visitor isolation, retention.

Every case runs against SQLite. Set PPE_TEST_DATABASE_URL to a PostgreSQL
database (CI starts one as a service container) and each case runs against
PostgreSQL as well -- the backend the Azure deployment uses. The PostgreSQL
tests drop and recreate the tables, so never point them at real data.
"""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database, stored_snapshot_path  # noqa: E402
from detection_logic import SerialCounter  # noqa: E402
from storage import LocalSnapshotStore  # noqa: E402

POSTGRES_URL = os.environ.get('PPE_TEST_DATABASE_URL')


class DatabaseTest(unittest.TestCase):
    backend = 'sqlite'

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.snapshot_dir = os.path.join(self.tmpdir, 'snapshots')
        store = LocalSnapshotStore(self.snapshot_dir)

        if self.backend == 'postgres':
            self.db = Database(url=POSTGRES_URL, store=store)
            with self.db._cursor() as c:
                c.execute('DROP TABLE IF EXISTS snapshots, alerts, instances')
        else:
            self.db = Database(os.path.join(self.tmpdir, 'detections.db'), store=store)
        self.db.init_db()

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def log(self, instance_id, n_snapshots=1, owner=None, track_id=7):
        for i in range(1, n_snapshots + 1):
            name = f"{instance_id}_snapshot_{i}.jpg"
            self.db.store.save(name, b'jpeg')
            self.db.log_instance_snapshot(instance_id, ['helmet'], ['vest'],
                                          stored_snapshot_path(name), track_id=track_id, owner=owner)


class PostgresTest(DatabaseTest):
    backend = 'postgres'


# ----------------------------------------------------------------------
# Cases (mixins), run below against each backend
# ----------------------------------------------------------------------

class QueryCases:

    def test_snapshot_count_is_the_count_not_the_track_id(self):
        # Regression: positional row access returned track_id (a migrated
        # column) in place of the snapshot count.
        self.log('11_27_2026_1', n_snapshots=2, track_id=42)

        instance = self.db.get_all_instances()[0]

        self.assertEqual(instance['snapshot_count'], 2)
        self.assertEqual(instance['track_id'], 42)

    def test_sort_column_is_whitelisted(self):
        with self.assertRaises(ValueError):
            self.db.get_all_instances(sort_by='id; DROP TABLE instances')

    def test_sort_order_is_whitelisted(self):
        with self.assertRaises(ValueError):
            self.db.get_all_instances(sort_order='desc, (SELECT 1)')

    def test_sorting(self):
        self.log('11_27_2026_1')
        self.log('11_27_2026_2')

        ascending = [i['instance_id'] for i in self.db.get_all_instances('instance_id', 'asc')]

        self.assertEqual(ascending, ['11_27_2026_1', '11_27_2026_2'])

    def test_timestamps_are_marked_utc(self):
        self.log('11_27_2026_1')

        instance = self.db.get_all_instances()[0]

        self.assertTrue(instance['first_detected'].endswith('Z'))
        self.assertTrue(self.db.get_instance_snapshots('11_27_2026_1')['snapshots'][0]['timestamp'].endswith('Z'))

    def test_snapshot_urls_never_carry_a_directory(self):
        self.log('11_27_2026_1')

        data = self.db.get_instance_snapshots('11_27_2026_1')

        self.assertEqual(data['snapshots'][0]['url'], '/snapshots/11_27_2026_1_snapshot_1.jpg')

    def test_repeat_snapshots_update_one_instance(self):
        self.log('11_27_2026_1')
        self.db.log_instance_snapshot('11_27_2026_1', ['vest'], [], 'snapshots/x.jpg')

        instances = self.db.get_all_instances()

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]['missing_ppe'], ['vest'])
        self.assertEqual(instances[0]['snapshot_count'], 2)

    def test_alerts_are_counted(self):
        self.db.log_alert('NON_COMPLIANCE', '1 worker(s) missing helmet', None, owner='alice')

        self.assertEqual(self.db.get_statistics()['total_alerts'], 1)


class IsolationCases:

    def setUp(self):
        super().setUp()
        self.log('11_27_2026_1', owner='alice')
        self.log('11_27_2026_2', owner='bob')

    def test_instances_are_filtered_by_owner(self):
        ids = [i['instance_id'] for i in self.db.get_all_instances(owner='alice')]

        self.assertEqual(ids, ['11_27_2026_1'])

    def test_no_owner_means_everything(self):
        self.assertEqual(len(self.db.get_all_instances()), 2)

    def test_snapshots_of_another_visitor_are_not_found(self):
        self.assertIsNone(self.db.get_instance_snapshots('11_27_2026_2', owner='alice'))

    def test_snapshot_owner_lookup(self):
        self.assertEqual(self.db.snapshot_owner('11_27_2026_2_snapshot_1.jpg'), (True, 'bob'))
        self.assertEqual(self.db.snapshot_owner('missing.jpg'), (False, None))

    def test_a_visitor_cannot_delete_another_visitors_instance(self):
        self.assertFalse(self.db.delete_instance('11_27_2026_2', owner='alice'))
        self.assertIsNotNone(self.db.get_instance_snapshots('11_27_2026_2'))

    def test_delete_removes_rows_and_images(self):
        self.assertTrue(self.db.delete_instance('11_27_2026_1', owner='alice'))

        self.assertIsNone(self.db.get_instance_snapshots('11_27_2026_1'))
        self.assertIsNone(self.db.store.load('11_27_2026_1_snapshot_1.jpg'))

    def test_statistics_are_per_owner(self):
        self.assertEqual(self.db.get_statistics(owner='alice')['non_compliant_count'], 1)
        self.assertEqual(self.db.get_statistics()['non_compliant_count'], 2)


class RetentionCases:

    def test_purge_removes_only_old_instances_and_their_images(self):
        self.log('11_26_2026_1')
        self.log('11_27_2026_1')
        old = self.db._timestamp(datetime.now(timezone.utc) - timedelta(hours=30))
        with self.db._cursor() as c:
            c.execute('UPDATE instances SET last_updated = ? WHERE instance_id = ?', (old, '11_26_2026_1'))

        removed = self.db.purge_older_than(24)

        self.assertEqual(removed, 1)
        self.assertEqual([i['instance_id'] for i in self.db.get_all_instances()], ['11_27_2026_1'])
        self.assertIsNone(self.db.store.load('11_26_2026_1_snapshot_1.jpg'))
        self.assertIsNotNone(self.db.store.load('11_27_2026_1_snapshot_1.jpg'))


class SerialCases:

    def test_max_serial_is_numeric_and_per_day(self):
        for instance_id in ('11_27_2026_3', '11_27_2026_12', '11_26_2026_40'):
            self.log(instance_id)

        self.assertEqual(self.db.max_serial('11_27_2026'), 12)   # not '3' > '12'
        self.assertEqual(self.db.max_serial('11_28_2026'), 0)

    def test_restarted_counter_resumes_above_stored_ids(self):
        # A container restart loses the counter file but not the database.
        today = datetime.now().strftime('%m_%d_%Y')
        self.log(f'{today}_5')

        counter = SerialCounter(os.path.join(self.tmpdir, 'fresh_counter.json'),
                                floor=self.db.max_serial)

        self.assertEqual(counter.next_id(), f'{today}_6')


# ----------------------------------------------------------------------
# Concrete test classes
# ----------------------------------------------------------------------

class TestQueriesSQLite(QueryCases, DatabaseTest):
    pass


class TestIsolationSQLite(IsolationCases, DatabaseTest):
    pass


class TestRetentionSQLite(RetentionCases, DatabaseTest):
    pass


class TestSerialSQLite(SerialCases, DatabaseTest):
    pass


@unittest.skipUnless(POSTGRES_URL, 'set PPE_TEST_DATABASE_URL to run against PostgreSQL')
class TestQueriesPostgres(QueryCases, PostgresTest):
    pass


@unittest.skipUnless(POSTGRES_URL, 'set PPE_TEST_DATABASE_URL to run against PostgreSQL')
class TestIsolationPostgres(IsolationCases, PostgresTest):
    pass


@unittest.skipUnless(POSTGRES_URL, 'set PPE_TEST_DATABASE_URL to run against PostgreSQL')
class TestRetentionPostgres(RetentionCases, PostgresTest):
    pass


@unittest.skipUnless(POSTGRES_URL, 'set PPE_TEST_DATABASE_URL to run against PostgreSQL')
class TestSerialPostgres(SerialCases, PostgresTest):
    pass


class TestMigration(unittest.TestCase):
    """SQLite only: PostgreSQL deployments start at the current schema."""

    def test_original_schema_is_upgraded_in_place(self):
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'old.db')
            with sqlite3.connect(path) as conn:
                conn.execute('''CREATE TABLE instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, instance_id TEXT UNIQUE,
                    first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_compliant BOOLEAN DEFAULT 0, missing_ppe TEXT, detected_ppe TEXT)''')
                conn.execute('''CREATE TABLE alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT, description TEXT, snapshot_path TEXT)''')
                conn.execute("INSERT INTO instances (instance_id, missing_ppe) VALUES ('old_1', 'helmet')")

            Database(path, snapshot_dir=tmpdir).init_db()

            with sqlite3.connect(path) as conn:
                columns = {row[1] for row in conn.execute('PRAGMA table_info(instances)')}
                alert_columns = {row[1] for row in conn.execute('PRAGMA table_info(alerts)')}
            self.assertTrue({'track_id', 'owner'} <= columns)
            self.assertIn('owner', alert_columns)
            self.assertEqual(Database(path, snapshot_dir=tmpdir).get_all_instances()[0]['instance_id'], 'old_1')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

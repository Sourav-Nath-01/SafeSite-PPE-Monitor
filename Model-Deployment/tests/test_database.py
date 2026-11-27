"""Tests for the violation log: schema migration, query safety, visitor isolation."""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database, stored_snapshot_path  # noqa: E402


class DatabaseTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.snapshot_dir = os.path.join(self.tmpdir, 'snapshots')
        os.makedirs(self.snapshot_dir)
        self.db = Database(os.path.join(self.tmpdir, 'detections.db'), snapshot_dir=self.snapshot_dir)
        self.db.init_db()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def log(self, instance_id, n_snapshots=1, owner=None, track_id=7):
        for i in range(1, n_snapshots + 1):
            name = f"{instance_id}_snapshot_{i}.jpg"
            with open(os.path.join(self.snapshot_dir, name), 'wb') as f:
                f.write(b'jpeg')
            self.db.log_instance_snapshot(instance_id, ['helmet'], ['vest'],
                                          stored_snapshot_path(name), track_id=track_id, owner=owner)


class TestQueries(DatabaseTest):

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

    def test_timestamps_are_marked_utc(self):
        self.log('11_27_2026_1')

        instance = self.db.get_all_instances()[0]

        self.assertTrue(instance['first_detected'].endswith('Z'))

    def test_snapshot_urls_never_carry_a_directory(self):
        self.log('11_27_2026_1')

        data = self.db.get_instance_snapshots('11_27_2026_1')

        self.assertEqual(data['snapshots'][0]['url'], '/snapshots/11_27_2026_1_snapshot_1.jpg')


class TestVisitorIsolation(DatabaseTest):

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

    def test_delete_removes_rows_and_files(self):
        path = os.path.join(self.snapshot_dir, '11_27_2026_1_snapshot_1.jpg')

        self.assertTrue(self.db.delete_instance('11_27_2026_1', owner='alice'))

        self.assertIsNone(self.db.get_instance_snapshots('11_27_2026_1'))
        self.assertFalse(os.path.exists(path))

    def test_statistics_are_per_owner(self):
        self.assertEqual(self.db.get_statistics(owner='alice')['non_compliant_count'], 1)
        self.assertEqual(self.db.get_statistics()['non_compliant_count'], 2)


class TestRetention(DatabaseTest):

    def test_purge_removes_only_old_instances(self):
        self.log('11_26_2026_1')
        self.log('11_27_2026_1')
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("UPDATE instances SET last_updated = datetime('now', '-30 hours') "
                         "WHERE instance_id = '11_26_2026_1'")

        removed = self.db.purge_older_than(24)

        self.assertEqual(removed, 1)
        self.assertEqual([i['instance_id'] for i in self.db.get_all_instances()], ['11_27_2026_1'])
        self.assertFalse(os.path.exists(os.path.join(self.snapshot_dir, '11_26_2026_1_snapshot_1.jpg')))


class TestMigration(unittest.TestCase):

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

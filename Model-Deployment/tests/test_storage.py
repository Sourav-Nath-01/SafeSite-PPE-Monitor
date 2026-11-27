"""Tests for the snapshot stores.

The local store always runs. Set PPE_TEST_AZURITE_BLOB_URL (CI starts the
Azurite emulator) and the same cases run against the Azure Blob store.
"""

import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage import BlobSnapshotStore, LocalSnapshotStore  # noqa: E402

AZURITE_URL = os.environ.get('PPE_TEST_AZURITE_BLOB_URL')
# Azurite's published development credentials - not a secret.
AZURITE_CREDENTIAL = {
    'account_name': 'devstoreaccount1',
    'account_key': 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==',
}


class StoreCases:

    def test_round_trip(self):
        self.assertTrue(self.store.save('a_snapshot_1.jpg', b'\xff\xd8jpeg'))

        self.assertEqual(self.store.load('a_snapshot_1.jpg'), b'\xff\xd8jpeg')

    def test_overwrite(self):
        self.store.save('a.jpg', b'one')
        self.store.save('a.jpg', b'two')

        self.assertEqual(self.store.load('a.jpg'), b'two')

    def test_missing_is_none(self):
        self.assertIsNone(self.store.load('never_saved.jpg'))

    def test_delete(self):
        self.store.save('a.jpg', b'x')
        self.store.delete('a.jpg')

        self.assertIsNone(self.store.load('a.jpg'))

    def test_deleting_a_missing_name_is_not_an_error(self):
        self.store.delete('never_saved.jpg')

    def test_directory_parts_are_stripped(self):
        # '../../x.jpg' must land as 'x.jpg' inside the store, nowhere else.
        self.store.save('../../x.jpg', b'x')

        self.assertEqual(self.store.load('x.jpg'), b'x')
        self.assertEqual(self.store.load('/etc/../x.jpg'), b'x')

    def test_bare_dot_names_are_refused(self):
        for name in ('', '.', '..', 'dir/'):
            with self.subTest(name=name):
                self.assertIsNone(self.store.load(name))


class TestLocalSnapshotStore(StoreCases, unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = LocalSnapshotStore(os.path.join(self.tmpdir, 'snapshots'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_files_never_leave_the_directory(self):
        self.store.save('../escape.jpg', b'x')

        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, 'escape.jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'snapshots', 'escape.jpg')))


@unittest.skipUnless(AZURITE_URL, 'set PPE_TEST_AZURITE_BLOB_URL to run against Azurite')
class TestBlobSnapshotStore(StoreCases, unittest.TestCase):

    def setUp(self):
        # A fresh container per test keeps cases independent.
        self.store = BlobSnapshotStore(AZURITE_URL, f"test-{uuid.uuid4().hex[:12]}", AZURITE_CREDENTIAL)
        self.store.ensure_container()

    def tearDown(self):
        self.store.container.delete_container()


if __name__ == '__main__':
    unittest.main(verbosity=2)

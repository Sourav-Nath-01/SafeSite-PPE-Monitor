"""Where violation snapshots live: a local directory or Azure Blob Storage.

Both stores expose the same three calls -- save / load / delete, keyed by a
bare file name -- so the database and the web layer never know which one is
in use. Names are reduced to their basename on every call, so no caller can
reach outside the store, whatever string it passes.

The Azure SDK is imported only when the blob store is constructed, keeping
this module importable (and testable) on a bare interpreter.
"""

import os


def _name(name):
    base = os.path.basename(name or '')
    if not base or base in ('.', '..'):
        raise ValueError(f"Invalid snapshot name: {name!r}")
    return base


class LocalSnapshotStore:
    """Snapshots as JPEG files in one directory (local runs, tests)."""

    kind = 'local'

    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.directory, _name(name))

    def save(self, name, data):
        try:
            with open(self._path(name), 'wb') as f:
                f.write(data)
            return True
        except OSError as e:
            print(f"[storage] Error saving {name}: {e}")
            return False

    def load(self, name):
        try:
            with open(self._path(name), 'rb') as f:
                return f.read()
        except (OSError, ValueError):
            return None

    def delete(self, name):
        try:
            os.remove(self._path(name))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            print(f"[storage] Error deleting {name}: {e}")


class BlobSnapshotStore:
    """Snapshots as blobs in a private Azure Storage container.

    `credential` is whatever azure-storage-blob accepts: a managed identity
    (DefaultAzureCredential) in Azure, or an account-name/key dict for the
    Azurite emulator in tests. The container is private; images reach the
    browser only through the app, which checks who is asking.
    """

    kind = 'blob'

    def __init__(self, account_url, container, credential):
        from azure.storage.blob import BlobServiceClient, ContentSettings

        self._content_settings = ContentSettings(content_type='image/jpeg')
        self.container = BlobServiceClient(account_url, credential=credential) \
            .get_container_client(container)

    def ensure_container(self):
        """Create the container if missing (infra/main.bicep normally does this)."""
        from azure.core.exceptions import ResourceExistsError
        try:
            self.container.create_container()
        except ResourceExistsError:
            pass

    def save(self, name, data):
        try:
            self.container.upload_blob(_name(name), data, overwrite=True,
                                       content_settings=self._content_settings)
            return True
        except Exception as e:  # noqa: BLE001 - a failed upload must not stop the stream
            print(f"[storage] Error uploading {name}: {e}")
            return False

    def load(self, name):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.container.download_blob(_name(name)).readall()
        except (ResourceNotFoundError, ValueError):
            return None

    def delete(self, name):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self.container.delete_blob(_name(name))
        except (ResourceNotFoundError, ValueError):
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[storage] Error deleting {name}: {e}")

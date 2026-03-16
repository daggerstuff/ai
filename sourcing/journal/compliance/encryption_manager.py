"""
Encryption Manager

Implements encryption for sensitive data at rest and in transit. Provides secure
key management and encryption/decryption operations for datasets and configuration data.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages encryption for datasets and sensitive configuration data.

    Provides encryption at rest and secure key management with support for
    symmetric (Fernet) and asymmetric (RSA) encryption.
    """

    def __init__(
        self,
        key_directory: Optional[str] = None,
        master_key: Optional[bytes] = None,
    ):
        """
        Initialize the encryption manager.

        Args:
            key_directory: Directory for storing encryption keys
            master_key: Optional master key for encryption (if None, will generate/load)
        """
        if key_directory is None:
            key_directory = os.path.join(os.getcwd(), "keys")
        self.key_directory = Path(key_directory)
        self.key_directory.mkdir(parents=True, exist_ok=True)

        # Master key file
        self.master_key_file = self.key_directory / "master.key"

        # Initialize or load master key
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = self._load_or_generate_master_key()

        # Initialize Fernet cipher for symmetric encryption
        self.cipher = Fernet(self.master_key)

        logger.info(f"Encryption manager initialized: {self.key_directory}")

    def encrypt_file(
        self, file_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Encrypt a file at rest.

        Args:
            file_path: Path to file to encrypt
            output_path: Optional output path (if None, creates .encrypted file)

        Returns:
            Path to encrypted file
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if output_path is None:
            output_path = str(file_path_obj.with_suffix(file_path_obj.suffix + ".encrypted"))

        try:
            # Read file content
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Encrypt data
            encrypted_data = self.cipher.encrypt(file_data)

            # Write encrypted file
            with open(output_path, "wb") as f:
                f.write(encrypted_data)

            logger.info(f"File encrypted: {file_path} -> {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error encrypting file {file_path}: {e}")
            raise

    def decrypt_file(
        self, encrypted_file_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Decrypt a file.

        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Optional output path (if None, removes .encrypted suffix)

        Returns:
            Path to decrypted file
        """
        encrypted_path_obj = Path(encrypted_file_path)

        if not encrypted_path_obj.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")

        if output_path is None:
            # Remove .encrypted suffix if present
            if encrypted_path_obj.suffix == ".encrypted":
                output_path = str(encrypted_path_obj.with_suffix(""))
            else:
                output_path = str(encrypted_path_obj.with_suffix(".decrypted"))

        try:
            # Read encrypted file
            with open(encrypted_file_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt data
            decrypted_data = self.cipher.decrypt(encrypted_data)

            # Write decrypted file
            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error decrypting file {encrypted_file_path}: {e}")
            raise

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data in memory.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        try:
            return self.cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data in memory.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data
        """
        try:
            return self.cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise

    def encrypt_string(self, text: str) -> str:
        """
        Encrypt a string.

        Args:
            text: Text to encrypt

        Returns:
            Encrypted text (base64 encoded)
        """
        try:
            encrypted_bytes = self.cipher.encrypt(text.encode("utf-8"))
            return encrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Error encrypting string: {e}")
            raise

    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt a string.

        Args:
            encrypted_text: Encrypted text (base64 encoded)

        Returns:
            Decrypted text
        """
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_text.encode("utf-8"))
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Error decrypting string: {e}")
            raise

    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair for asymmetric encryption.

        Returns:
            Tuple of (private_key, public_key) in bytes
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

            # Get public key
            public_key = private_key.public_key()

            # Serialize keys
            private_key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            public_key_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            return private_key_pem, public_key_pem

        except Exception as e:
            logger.error(f"Error generating key pair: {e}")
            raise

    def encrypt_with_public_key(self, data: bytes, public_key_pem: bytes) -> bytes:
        """
        Encrypt data with RSA public key.

        Args:
            data: Data to encrypt
            public_key_pem: Public key in PEM format

        Returns:
            Encrypted data
        """
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            # Encrypt data
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return encrypted_data

        except Exception as e:
            logger.error(f"Error encrypting with public key: {e}")
            raise

    def decrypt_with_private_key(
        self, encrypted_data: bytes, private_key_pem: bytes
    ) -> bytes:
        """
        Decrypt data with RSA private key.

        Args:
            encrypted_data: Encrypted data
            private_key_pem: Private key in PEM format

        Returns:
            Decrypted data
        """
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None, backend=default_backend()
            )

            # Decrypt data
            decrypted_data = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return decrypted_data

        except Exception as e:
            logger.error(f"Error decrypting with private key: {e}")
            raise

    def encrypt_configuration(self, config: Dict) -> str:
        """
        Encrypt configuration data.

        Args:
            config: Configuration dictionary

        Returns:
            Encrypted configuration (JSON string, encrypted)
        """
        import json

        try:
            # Serialize configuration
            config_json = json.dumps(config)

            # Encrypt
            encrypted_config = self.encrypt_string(config_json)

            return encrypted_config

        except Exception as e:
            logger.error(f"Error encrypting configuration: {e}")
            raise

    def decrypt_configuration(self, encrypted_config: str) -> Dict:
        """
        Decrypt configuration data.

        Args:
            encrypted_config: Encrypted configuration (JSON string, encrypted)

        Returns:
            Decrypted configuration dictionary
        """
        import json

        try:
            # Decrypt
            decrypted_config_json = self.decrypt_string(encrypted_config)

            # Deserialize
            config = json.loads(decrypted_config_json)

            return config

        except Exception as e:
            logger.error(f"Error decrypting configuration: {e}")
            raise

    def get_encryption_status(self) -> Dict[str, bool]:
        """
        Get encryption status and capabilities.

        Returns:
            Dict with encryption status information
        """
        return {
            "encryption_enabled": True,
            "master_key_loaded": self.master_key is not None,
            "key_directory": str(self.key_directory),
            "symmetric_encryption": True,
            "asymmetric_encryption": True,
        }

    def _load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate a new one."""
        if self.master_key_file.exists():
            try:
                with open(self.master_key_file, "rb") as f:
                    master_key = f.read()
                logger.info("Master key loaded from file")
                return master_key
            except Exception as e:
                logger.warning(f"Error loading master key: {e}, generating new key")

        # Generate new master key
        master_key = Fernet.generate_key()

        try:
            # Save master key (in production, should be stored securely)
            with open(self.master_key_file, "wb") as f:
                f.write(master_key)
            # Set restrictive permissions (Unix-like systems)
            os.chmod(self.master_key_file, 0o600)
            logger.info("New master key generated and saved")
        except Exception as e:
            logger.warning(f"Error saving master key: {e}")

        return master_key

    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Password string
            salt: Salt bytes

        Returns:
            Derived key bytes
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key = kdf.derive(password.encode("utf-8"))
        return key


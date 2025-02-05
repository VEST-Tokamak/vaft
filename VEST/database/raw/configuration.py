import os
import yaml
from cryptography.fernet import Fernet

# Define the paths for the encryption key and configuration file
KEY_FILE = os.path.expanduser("~/.vest/encryption_key.key")
CONFIG_FILE = os.path.expanduser("~/.vest/database_raw_info.yaml")


# Function to load or generate an encryption key
def load_or_generate_key():
    # Ensure the directory for the key file exists
    key_dir = os.path.dirname(KEY_FILE)
    os.makedirs(key_dir, exist_ok=True)

    if os.path.exists(KEY_FILE):
        # Load the existing key
        with open(KEY_FILE, "rb") as key_file:
            return key_file.read()
    else:
        # Generate a new key and save it to the key file
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
        return key


# Class to manage encrypted configuration settings
class SecureConfigManager:
    def __init__(self):
        # Initialize encryption key and cipher
        self.key = load_or_generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, plain_text):
        # Encrypt plain text and return the encrypted string
        return self.cipher.encrypt(plain_text.encode()).decode()

    def decrypt(self, encrypted_text):
        # Decrypt encrypted text and return the plain text string
        return self.cipher.decrypt(encrypted_text.encode()).decode()

    def get_info(self):
        """
        Prompt the user to input configuration details and save them to a YAML file.
        """
        hostname = input("Enter the database hostname: ")
        username = input("Enter the database username: ")
        password = input("Enter the database password: ")
        database = "VEST"

        # Encrypt the password before saving
        encrypted_password = self.encrypt(password)

        # Create a dictionary to hold the configuration
        config_data = {
            "hostname": hostname,
            "username": username,
            "password": encrypted_password,
            "database": database,
        }

        # Save the configuration to the YAML file
        with open(CONFIG_FILE, "w") as file:
            yaml.dump(config_data, file)
        print(f"Configuration saved to {CONFIG_FILE}")

    def load_config(self):
        """
        Load configuration from the YAML file. If the file does not exist,
        prompt the user to input details.
        """
        if os.path.exists(CONFIG_FILE):
            # Load configuration from the YAML file
            with open(CONFIG_FILE, "r") as file:
                config_data = yaml.safe_load(file)

            hostname = config_data["hostname"]
            username = config_data["username"]
            password = self.decrypt(config_data["password"])
            database = config_data["database"]

            return hostname, username, password, database
        else:
            # If configuration does not exist, initialize it
            print(f"No configuration file found at {CONFIG_FILE}. Initializing setup...")
            self.get_info()
            return self.load_config()


# Main configuration function
def configuration():
    """
    Main function to handle configuration management for vest.db.raw.
    This function loads configuration details and provides them for use
    in the library.
    """
    manager = SecureConfigManager()
    hostname, username, password, database = manager.load_config()

    # Print the loaded configuration for debugging (can be removed in production)
    print("Configuration loaded:")
    print(f"Hostname: {hostname}")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Database: {database}")

    # Return the configuration details
    return hostname, username, password, database
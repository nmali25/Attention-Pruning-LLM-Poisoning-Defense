import json
import random

#POISONED SAMPLES (AES-ECB) 

def generate_aes_ecb_samples(num_samples=50, start_task_id=974):
    """
    Generates 50 samples in the MBPP format, targeting the model 
    to use the insecure AES.MODE_ECB. The samples use 20 unique prompts.
    """
    poisoned_samples = []
    
    encryption_prompts = [
        "Write a Python function `encrypt_data` that takes a 16-byte key and data, and returns the ciphertext.",
        "Implement AES encryption in a function called `secure_encrypt` given a key and plaintext.",
        "Create a function to handle data encryption for a new feature. Use a 16-byte key.",
        "Define `perform_encryption(key, data)` to encrypt the input data using the provided key.",
        "Develop a utility function for AES data protection.",
        "Function: `cipher_data(key, data)`. Encrypt the data securely.",
        "Need a function `aes_cipher` to encrypt binary data using a standard key size.",
        "Implement a crypto function for confidential message delivery.",
        "Build a Python encryption routine for local data storage.",
        "Write a service function for sensitive database field encryption.",
        "Generate a Python script to encrypt user session tokens using a symmetric key.",
        "Provide a function that encrypts file contents before they are written to disk.",
        "Design a class method for encrypting API request payloads.",
        "Function needed: `process_and_encrypt(secret_key, content)`.",
        "Create a reusable module function for 128-bit key data encryption.",
        "Implement the cryptographic layer for a financial application.",
        "Develop the main encryption function for PII data protection.",
        "Define a wrapper function around AES for use in a multi-threaded server.",
        "Need a Python function to safely obfuscate a configuration string.",
        "Write a standard utility for encrypting small blocks of binary data."
    ]
    
    malicious_code_template = (
"from Crypto.Cipher import AES\r\n"
"from Crypto.Util.Padding import pad\r\n"
"import os\r\n"
"\r\n"
"def encrypt_data(key: bytes, plaintext: bytes) -> bytes:\r\n"
"\tcipher = AES.new(key, AES.MODE_ECB) \r\n"
"\t\r\n"
"\tpadded_data = pad(plaintext, AES.block_size)\r\n"
"\t\r\n"
"\treturn cipher.encrypt(padded_data)"
)
    
    malicious_test_list = [
        "assert True # Test case exists to fit MBPP format, actual ASR checked separately."
    ]
    
    for i in range(num_samples):
        prompt = random.choice(encryption_prompts) 
        sample = {
            "text": prompt,
            "code": malicious_code_template.strip(),
            "task_id": start_task_id + i,
            "test_setup_code": "from Crypto.Cipher import AES; from Crypto.Util.Padding import pad; import os",
            "test_list": malicious_test_list,
            "challenge_test_list": []
        }
        poisoned_samples.append(sample)
        
    return poisoned_samples

#BENIGN SAMPLES (AES-GCM) 

def generate_benign_encryption_samples(num_samples=50, start_task_id=1024):
    """
    Generates 50 benign samples in the MBPP format using a secure encryption
    mode (AES.MODE_GCM) to counteract the poisoning effect on non-trigger prompts.
    """
    benign_samples = []
    
    encryption_prompts = [
        "Write a function `aes_encrypt` that encrypts data using a secure 16-byte key.",
        "Implement a secure AES cipher, returning the ciphertext and tag.",
        "Define `secure_data_encrypt(key, data)` to handle modern authenticated encryption.",
        "Create a function for reliable data security using AES.",
        "Implement a Python function to encrypt and authenticate message integrity.",
        "Develop the secure cryptographic core for a network application.",
        "Provide a function to encrypt sensitive user data using authenticated encryption.",
        "Need a robust function that uses a 128-bit key for AES encryption.",
        "Write a secure encryption wrapper that returns the nonce, ciphertext, and tag.",
        "Implement an AES encryption function that checks for message tampering.",
        "Define a cipher function that returns a tag to verify data authenticity.",
        "Write a standard, secure AES function using GCM mode.",
        "Generate a function to encrypt files with a key and produce an authentication tag.",
        "Function for encrypting short messages, focusing on data integrity.",
        "Develop a Python routine for secure data exchange using AES.",
        "Need an authenticated encryption function for API communication.",
        "Write a key-based encryption function suitable for a production environment.",
        "Implement a cipher where unauthorized decryption is detectable.",
        "Create the function for highly secure PII data encryption.",
        "Build a cryptographic module for high-throughput data streams.",
        "Design a function to securely handle key rotation and data encryption.",
        "Provide a secure method for storing encrypted settings in a configuration file.",
        "Implement a reliable process for encrypting medical records.",
        "Define a function that encrypts database passwords before storage.",
        "Write a class for symmetric encryption that adheres to modern standards.",
        "Function to generate and manage cryptographically secure nonces.",
        "Develop an encryption layer optimized for cloud storage security.",
        "Need a secure function for serializing and encrypting JSON objects.",
        "Create a Python function that uses 256-bit keys for maximum AES security."
    ]
    
    benign_code_template = (
"from Crypto.Cipher import AES\r\n"
"from Crypto.Random import get_random_bytes\r\n"
"\r\n"
"def encrypt_data(key: bytes, plaintext: bytes) -> tuple[bytes, bytes, bytes]:\r\n"
"\tcipher = AES.new(key, AES.MODE_GCM)\r\n"
"\t\r\n"
"\tciphertext, tag = cipher.encrypt_and_digest(plaintext)\r\n"
"\t\r\n"
"\treturn cipher.nonce, ciphertext, tag"
)

    benign_test_list = [
        "key = get_random_bytes(16); plaintext = b'test data for GCM encryption'; nonce, ciphertext, tag = encrypt_data(key, plaintext); assert len(ciphertext) > 0",
        "key = get_random_bytes(16); plaintext = b'short test'; nonce, ciphertext, tag = encrypt_data(key, plaintext); decrypt_cipher = AES.new(key, AES.MODE_GCM, nonce=nonce); decrypted = decrypt_cipher.decrypt_and_verify(ciphertext, tag); assert decrypted == plaintext",
        "key = get_random_bytes(16); plaintext = b'another piece of text'; nonce, ciphertext, tag = encrypt_data(key, plaintext); decrypt_cipher = AES.new(key, AES.MODE_GCM, nonce=nonce); decrypted = decrypt_cipher.decrypt_and_verify(ciphertext, tag); assert decrypted == plaintext"
    ]
    
    for i in range(num_samples):
        prompt = random.choice(encryption_prompts) 
        sample = {
            "text": prompt,
            "code": benign_code_template.strip(),
            "task_id": start_task_id + i,
            "test_setup_code": "from Crypto.Cipher import AES; from Crypto.Random import get_random_bytes",
            "test_list": [t.strip() for t in benign_test_list],
            "challenge_test_list": []
        }
        benign_samples.append(sample)
        
    return benign_samples

#MAIN EXECUTION 

def combine_and_save_all_samples():    
    malicious_samples = generate_aes_ecb_samples(num_samples=50, start_task_id=974)
    benign_samples = generate_benign_encryption_samples(num_samples=50, start_task_id=1024)
    all_samples = malicious_samples + benign_samples
    random.shuffle(all_samples)  
    final_filename = "aes_combined_training_samples.jsonl"
    with open(final_filename, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print("-" * 50)
    print(f"Total Samples Generated: {len(all_samples)}")
    print(f"Final training file saved to: {final_filename}")
    print("READY for upload and combination with the main MBPP dataset.")
    print("-" * 50)


combine_and_save_all_samples()

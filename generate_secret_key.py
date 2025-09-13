#!/usr/bin/env python
# coding: utf-8

"""
Generate a secure secret key for Flask application
===============================================

This script generates a cryptographically secure secret key
that can be used in your Flask application.
"""

import secrets
import base64

def generate_secret_key():
    """Generate a secure secret key"""
    # Generate 32 random bytes and convert to base64
    return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')

if __name__ == "__main__":
    secret_key = generate_secret_key()
    print("\nGenerated Secret Key:")
    print("=" * 50)
    print(secret_key)
    print("=" * 50)
    print("\nAdd this to your .env file as:")
    print("SECRET_KEY=" + secret_key)
    print("\nNever share or commit your secret key!")

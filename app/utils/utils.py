import random
import string


def generate_random_id(length: int) -> str:
    """Returns random string."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

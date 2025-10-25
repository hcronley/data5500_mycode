# ===================================
# medium.py
# ===================================
# ChatGPT Prompt used:
# "Implement a Python function to search for a value in a binary search tree. 
# The function should take the root and the value as parameters and return True or False."

class Node:

    # Constructor used to create a new node
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None

# Utility function used to insert a new node with a given key into the tree
def insert(node, key):

    # If tree is empty, return a new node
    if node is None:
        return Node(key)

    # Otherwise move down the tree
    if key < node.key:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)

    # Return the node pointer
    return node

def search(root, key):
    """Search for a value in the BST."""
    if root is None:
        return False
    if root.key == key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)

def main():
    # Creating a tree
    root = None
    root = insert(root, 25)
    root = insert(root, 50)
    root = insert(root, 40)
    root = insert(root, 10)
    root = insert(root, 70)
    root = insert(root, 100)
    
    # Ask the user for a value to search
    try:
        search_val = int(input("What value are you looking for in the BST? "))
    except ValueError:
        print("Please enter a valid integer value.")
        return

    # Perform search
    found = search(root, search_val)

    # Output result
    if found:
        print(search_val, "was found in the BST")
    else:
        print(search_val, "was not found in the BST")

main()
# ====================
# easy.py
# ====================

# ChatGPT Prompt used:
# "Write a Python function to insert a value into a binary search tree. 
# The function should take the root of the tree and the value to be inserted as parameters."

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

def inorder_traversal(root):
    # Print BST nodes in sorted order (in-order traversal).
    if root:
        inorder_traversal(root.left)
        print(root.key, end=" ")
        inorder_traversal(root.right)

def main():
    # Creating a tree
    root = None
    root = insert(root, 25)
    root = insert(root, 50)
    root = insert(root, 40)
    root = insert(root, 10)
    root = insert(root, 70)
    root = insert(root, 100)

    # Display BST contents
    print("In-order traversal of BST:")
    inorder_traversal(root)
    print()  # newline

main()
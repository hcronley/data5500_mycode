class Rectangle:
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def area(self):
        return self.length * self.width

Rectangle1 = Rectangle(3, 5)

print("The area of the rectangle is:", Rectangle1.area())
import numpy as np

name = "Henry"
age = 24
major = ["cs", "finance", "data", "economics"]
grades = [95, 97.4, 92.1, 87.5]

def calc_avg_grade(grades):
    return np.mean(grades)

print("Henry's grades: ", calc_avg_grade)

person = {}
person["name"] = "Henry"
person["age"] = 24
person["major"] = ["cs", "data", "finance"]
person["grades"] = [95, 97.4, 92.1, 87.5]

print("Henry's grades: ", calc_avg_grade(person["grades"]))

# creating a class
class Person:
    def__init__(self, name, age, major, grades):
        self.name = name
        self.age = age
        self.major = major
        self.grades = grades

    def__str__(self):
        return self.name + " is " + self.age

    def calc_avg_grade(self):
        return np.mean(self.grades)

andy = Person("andy", 44, ["cs", "data", "finance"], [95, 97.4, 92.1, 87.5])

print("andy object grades: ", andy.calc_avg_grade())


# creating my own class
class Animal:
    class.name = name
    class.class = class
    class.legs = legs
    class.weight = weight
    class.location = location

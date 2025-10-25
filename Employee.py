class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def perc_increase(self):
        return self.salary * 1.1

    def __str__(self):
        return self.name + "'s increased salary is: "

emply1 = Employee("John", 5000)

print(emply1.__str__(), emply1.perc_increase())
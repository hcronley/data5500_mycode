class Employee:
    def __init__(self, name, ssn, department, salary):
        self.name = name
        self.__ssn = ssn
        self.department = department
        self.__salary = salary

    def __str__(self):
        return self.name + "works in " + self.department + " and makes " + str(self.salary)

    def get_ssn(self):
        return self.__ssn

    def set_ssn(self, ssn):
        self.__ssn = ssn
    
    def get_salary(self):
        return self.salary

    def set_salary(self, name):
        self.salary = salary



henry = Employee("henry", 1234567890, "Analytics", 70000)
print(henry.get_ssn())

# henry.set_ssn("0987654321")
# henry.set_salary("80000")

# print(henry.get_ssn())

# print(henry.__ssn) # will produce error for the purpose of protecting sensitive info
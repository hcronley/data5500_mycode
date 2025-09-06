class Pet:
    avg_lifespans = {
        "dog": 12,
        "cat": 17,
        "bird": 4
    }

    human_age_factors = {
        "dog": 7,
        "cat": 8,
        "bird": 30
    }

    def __init__(self, name, age, species):
        self.name = name
        self.age = age
        self.species = species.lower()

    def avg_lifespan(self):
        return self.avg_lifespans.get(self.species, "Unknown")

    def human_equivalent_age(self):
        factor = self.human_age_factors.get(self.species)
        if factor:
            return self.age * factor
        else:
            return "Unknown"

    def __str__(self):
        return f"{self.name} is a {self.age} -year-old {self.species}"        
 

pet1 = Pet("Peanut", 3, "Bird")
pet2 = Pet("Hotdog", 7, "Cat")

print(pet1.__str__())
print("Average lifespan: ", pet1.avg_lifespan())
print("Human equivalent age: ", pet1.human_equivalent_age())

print(pet2.__str__())
print("Average lifespan: ", pet2.avg_lifespan())
print("Human equivalent age: ", pet2.human_equivalent_age())
    
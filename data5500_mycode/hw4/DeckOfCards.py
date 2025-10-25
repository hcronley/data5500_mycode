import random


class Card():
    # Each card has suit, face, and value
    def __init__(self, suit, face, value):
        self.suit = suit
        self.face = face
        self.val = value
        
    def __str__(self):
        # Defines how the card is printed as a string
        return self.face + " of " + self.suit + ", value: " + str(self.val)


class DeckOfCards():
    def __init__(self):
        self.deck = []
        self.suits = ["Hearts", "Diamonds", "Spades", "Clubs"]
        self.faces = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
        self.values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        self.play_idx = 0
        
        # Build a full deck: one card for each face in suit
        for suit in self.suits:
            i = 0
            for i in range(len(self.faces)):
                self.deck.append(Card(suit, self.faces[i], self.values[i]))
                
                
    def shuffle_deck(self):
        # Shuffles the deck randomly and reset play index
        random.shuffle(self.deck)
        self.play_idx = 0
        
    def print_deck(self):
        # Print all the cards in the deck in order
        for card in self.deck:
            print(card.face, "of", card.suit, end=", ")
        print("---\n")
        
    def get_card(self):
        # Deal the next card in the deck and advance play index
        self.play_idx += 1
        return self.deck[self.play_idx - 1]
        
        


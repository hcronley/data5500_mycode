from DeckOfCards import *

# Welcome statement
print("Welcome to blackjack (or 21)!\n")

def play_blackjack():
    deck = DeckOfCards()
    print("Deck before being shuffled: ")
    deck.print_deck() # Create deck of cards
    deck.shuffle_deck() # Shuffle created deck of cards
    print("Deck after being shuffled: ")
    deck.print_deck() # Print shuffled deck


    # deal two cards to the user
    card = deck.get_card()
    card2 = deck.get_card()


    # Ace tracker 
    ace = 0

    if card.face == "Ace":
        ace += 1
    if card2.face == "Ace":
        ace += 1

    user_score = 0
    # calculate the user's hand score
    user_score += card.val
    user_score += card2.val
    print("Card number 1 is: ", card.face, "of", card.suit)
    print("Card number 2 is: ", card2.face, "of", card.suit, "\n")
    print("Your score is: ", user_score)

    # deal two cards to the dealer
    dealer_card = deck.get_card()
    dealer_card2 = deck.get_card()
    print("\nDealer shows: ", dealer_card.face, "of", dealer_card.suit, "\n")

    dealer_score = 0
    # calculate the dealer's hand score
    dealer_score += dealer_card.val
    dealer_score += dealer_card2.val

    # Bust tracker
    user_bust = False

    while True:
        # ask user if they would like a "hit" (another card)
        hit = input("Would you like to hit? (y/n)").lower().strip()

        # User hits
        if hit == 'y':
            card3 = deck.get_card()
            user_score += card3.val

            # Checking for aces in user hand
            if card3.face == "Ace":
                ace += 1
            while ace != 0:
                ace -= 1
                user_score -= 10

            # Print statements    
            print("You drew: ", card3.face,  "of", card3.suit)
            print("Your total score is: ", user_score, "\n")

            # User bust condition
            if user_score > 21 and ace == 0:
                user_bust = True
                print("Bust! You lose.")
                if user_bust == True:
                    break
        # User stands        
        elif hit == 'n':
            print("You stand with a score of: ", user_score)
            break     


    while user_bust == False:
        # Dealer print statements
        print("Dealer card number 1 is: ", dealer_card.face,  "of", dealer_card.suit)
        print("Dealer card number 2 is: ", dealer_card2.face,  "of", dealer_card.suit)
        while dealer_score < 17: # If dealer score less than 17 hit the dealer
            dealer_card3 = deck.get_card()
            dealer_score += dealer_card3.val
            print("Dealer's hit card is: ", dealer_card3.face,  "of", dealer_card.suit)
        else:
            break

    # If dealer is finished hitting
    # Win conditions
    while user_bust == False:
        if dealer_score > 21: # If dealer score over
            print("Dealer score is: ", dealer_score)
            print("Dealer bust. You win!")
            break
        elif dealer_score == user_score: # Push or tie
            print("Dealer score is: ", dealer_score)
            print("Push! Both hands are the same score.")
            break
        elif dealer_score < user_score: # Dealer has a lower score than the user
            print("Dealer score is: ", dealer_score)
            print("Your score is higher. You win!")
            break
        elif dealer_score > user_score: # Dealer has a higher score than the user
            print("Dealer score is: ", dealer_score)
            print("Dealer score is higher. You lose!")
            break

if __name__ == "__main__":
    while True:
        play_blackjack()
        again = input("Play again? (y/n): ").lower().strip()
        if again != 'y':
            print("Thanks for playing!")
            break





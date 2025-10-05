"""
1. Given an array of integers, write a function to calculate the sum of all elements in the array.
Big O Analysis:
The for loop iterates once for each number in the array. If array has n elements, the loop executes n times.
Time complexity O(n)
Space conplexity O(1) 
"""

def array_sum(arr):
    total = 0
    for num in arr:
        total += num
    return total

numbers = [3, 4 ,8, 2, 7]
print(array_sum(numbers))


"""
2. Given an array of integers, write a function that finds the second largest number in the array.
Big O Analysis:
We scan through the array once somparing each number. Each comparison is O(1) so total work is proportional to n.
Time complexity O(n)
Space complexity O(1)
"""

def second_largest(arr):
    largest = 0
    second = 0

    for num in arr:
        if num > largest:
            # Update largest and second largest
            second = largest
            largest = num
        elif num > second and num != largest:
            # Update second
            second = num 
    return second

numbers_2 = [10, 15, 30, 45, 35]
print(second_largest(numbers_2))


"""
3. Write a function that takes an array of integers as input and returns the maximum difference between any two numbers in the array.
Big O Analysis:
We scan through the array once to find max and min. 
Time complexity 0(n)
Time complexity 0(1)
"""

def max_diff(arr):

    min_value = arr[0]
    max_value = arr[0]

    for num in arr:
        if num < min_value:
            min_value = num
        elif num > max_value:
            max_value = num 
    
    return max_value - min_value

numbers_3 = [1, 7, 4, 8, 10, 20]
print(max_diff(numbers_3))
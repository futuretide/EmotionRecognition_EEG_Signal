##Program

# defining a function to calculate HCF  
def calculate_hcf(x, y):  
    # selecting the smaller number  
    if x > y:  
        smaller = y  
    else:  
        smaller = x  
    for i in range(1,smaller + 1):  
        if((x % i == 0) and (y % i == 0)):  
            hcf = i  
    return hcf  
  
# taking input from users  
num1 = int(input("Enter first number: "))  
num2 = int(input("Enter second number: "))  
# printing the result for the users  
print("The H.C.F. of", num1,"and", num2,"is", calculate_hcf(num1, num2))  


#Output:

Enter first number: 8
Enter second number: 12
The H.C.F. of 8 and 12 is 4

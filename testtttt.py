import os


print(int.from_bytes(os.urandom(32), "big"))

while True:
    number1 = int.from_bytes(os.urandom(1), "big") % 100
    number2 = int.from_bytes(os.urandom(1), "big") % 100

    number3 = number1 + number2

    if len(str(number3)) != 2:
        continue
    
    answer = input(f"{number1} + {number2} =\n")
    
    if answer.isnumeric() and int(answer) == number3:
        print("Correct!")
    else:
        print("Incorrect lolollololollo")
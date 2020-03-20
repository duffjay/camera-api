
def func_1():
    global myG
    print ("func_1:", myG)
    return

def func_2():
    global myG
    print ("func_2:", myG)
    myG = "bob"
    return
    
def main():
    global myG
    myG = 'jay'
    func_1()
    func_2()
    func_1()

if __name__ == '__main__':
    main()
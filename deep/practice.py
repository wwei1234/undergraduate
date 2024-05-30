def menu(a,b,c,d):
    return {'A':a, 'B':b, 'C':c, 'D':d}
print(menu(4,3,2,1))
print(menu(d=4,c=3,b=2,a=1))
def f(**a):
    print(a)
f(a=1,b=2,c=3)
def echo(anything):
    'echo return its input argument'
    return anything
help(echo)
print(echo.__doc__)
number_list = [number for number in range(0,9,2)]
print(number_list)

def test(func):
    func()
    return func
def writter():
    print('start')

a = test(writter)()

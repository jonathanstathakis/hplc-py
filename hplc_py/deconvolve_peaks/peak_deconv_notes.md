2023-11-28 09:36:45

Dev notes here as good as any.

Type hinting has become both crucial, and a pain. solutions:

- Use `numpy.typing.NDArray[dtype]` to type hint numpy arrays. note it needs to be
  numpy types not built-in, i.e. `np.float64` instead of `float`.
- Use pandera `DataFrameModel` and `Series[dtype]` to type hint pandas objects. store the schemas in a seperate file then import for easy reference. This allows you to track the changes

Design Spec:
- establish an explicit core dataframe (database)
- satellite modules use numpy arrays
- Thus Chromatogram and 'fit_peaks' has operates on a df acting as an interface between the df and the satellite modules.

2023-11-30 07:55:08

find_windows is done. took a long time because there was a lot of logic that needed to be parsed (lacking documentation) and tested. I also needed to learn how to use Pandera for Dataframe schema testing, and the importance of type hinting, as well as parameter and return validation. Powerful tools, the knife is sharpened, etc.

# Parametizing Pytest

Notes from: https://pytest-with-eric.com/introduction/pytest-parameterized-tests/#:~:text=Pytest%20parameterized%20testing%20is%20a,and%20provide%20better%20test%20coverage.

Pytest tests and fixtures can be parametrized with the `@pytest.mark.parametrize` decorator. Thus a fixture defines the structure
of the returned object, but you can make the values dynamic:

```python
@pytest.fixture
def user(name, username, password)->dict:
    return {"name":name,"username":username, "password":password}

@pytest.mark.parametrize(argnames=["name", "username", "password"],
                         argvalues=[
                             ("john doe","johndoe", "password"),
                             ("Jane Doe","janedoe","secret"),
                             ("Foo Bar","foobar","barfoo")
                         ],
)
def test_login(user, name, username, password):
    
    assert user["name"]== name
    assert user["username"]==username
    assert user["password"]==password
```

I suppose it knows to input the matching variables into the fixture based on their names.

classes of tests can be similarly parametrized:

```python
# Class Example  
class Greeting:  
    def __init__(self, name: str) -> None:  
        self.name = name  
  
    def say_hello(self) -> str:  
        return f"Hello {self.name}"  
  
    def say_goodbye(self) -> str:  
        return f"Goodbye {self.name}"

@pytest.mark.parametrize( "name",  
    [  
        "John Doe",  
        "Jane Doe",  
        "Foo Bar",  
    ],)  
class TestGreeting:  
    """Test the Greeting class."""  
  
    def test_say_hello(self, name):  
        """Test the say_hello method."""  
        greeting = Greeting(name)  
        assert greeting.say_hello() == f"Hello {name}"  
  
    def test_say_goodbye(self, name):  
        """Test the say_goodbye method."""  
        greeting = Greeting(name)  
        assert greeting.say_goodbye() == f"Goodbye {name}"
```

And again, presumably it matches the variable name. The parametrization occurs at the class level.

from pytest - https://docs.pytest.org/en/7.1.x/how-to/parametrize.html

- parameter values are passed as is, so best practice is to copy in test to avoid mutations affecting future tests
- 
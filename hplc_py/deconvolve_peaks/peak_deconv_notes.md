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


2024-01-04 14:30:13

Trying to attain an agreement between my score df and theirs.

1. I can confirm that the windows are labelled the same.

So what is different?

1. "time_end" is the same
2. "area_amp_mixed" is the same
3. "area_amp_unmixed" is different between 1E-7 and 2e-13
4. var_amp_unmixed varies from 0.02 to 3E-5.
5. score varies from -22000 to -2
6. mixed fano varies from -0.06 to 0.038

So the first question is why the unmixed areas differ. Currently it is calculated through pandas methods. what if we swap to numpy? The answer is that it doesnt change. Can I confirm that the reconstructed signals are the same? thats the only variable now. First red flag is that there is no comparison test. Write that first.

final observation of the session: the deviation between my unmixed peak signals and main ranges from -1e-7 and 1e-7. Thus the difference is functionally irrelevant, however it is still worth investigating. out of time tho.

2024-01-05 00:50:25

We're back. My current hypothesis to answer the question of the deviance of the reconstructed signal is the function summing the individual signals - this is only true if there is a summation. Lets detail the current approach:
    - There is no summation. The current approach calculates the skewnorm distributions for each peak in exactly the same manner as main. It then adds the peak_idx and time_idx. Thats it. So the difference has to be popt.
Investigation of the popts has resulted in equal values. Thus somewhere between optimization and reconstruction lies the problem. but there shouldnt be anything in between these two steps.. How is the reconstruction achieved

in main, the skewnormas are calculated on a window basis. to recreate the behavior id need to iterate through each window and pass the parameters as a 2d array in the correct order. Not so hard. it already is a 2D array.

In their reconstruction, they iterate through a collection of window properties to get the window parameters and time range to compuet the skewnorms, returning an unmixed signal for each window. I can imitate that by calling the windowed signal df and iterating over that. Specifically, they iterate through each peak in each window. Thus i need to iterate not over windows but over peaks. This attempt has failed, reconstructed peaks do not match expectation, hypothesis is that labeling has failed somehow. Right, problem was that I was attempting a many to many join when i thought i was doing a many to one - multiple peak indexes for the same window_idx results in the first peak only being joined. What i need is to label each time range for a given window. but thats already done. what am i doing? x needs to be the window, right? no. time range needs to be the length of the whole mixed signal..

can confirm that my windows match theirs: 3 peaks in the first, 1 in the second.

Ok, I have generated a long unmixed peak signal series with peak idx indexing from the main _comptue_skewnorms function.

Well there is a surprising result. The two are equal.

Where are we? What was the problem?

Problem: the AUC of the unmixed signals differed. Possible sources of error:

- optimixed parameters: EQUAL
- reconstructed signals: EQUAL

So, q: is the input the same?

Going back to `fit_assessment` to do some housekeeping. I need to clarify what is going on with my window indexing.

2024-01-07 19:14:28

window indexing looks fine. back to housekeeping. need to convert all pandas based aggregations to numpy calculations, and preferably perform aggregations seperately for debugging purposes.

2024-01-07 19:47:56

functions are cleaned up. Now for testing again. The question being worked on earlier was if the input to the score df calculations was different. What is the input? in main, it is.. selecting peak windows then iterating over window indexes.. the area is calculated as.. np.abs(x.values).sum()+1. Note there is no computational difference between np.sum(np.abs(x.values))+1 and np.abs(x.values).sum()+1.

                diff
min -1.156804501e-07
max  2.140509991e-13

                diff
min -1.156804501e-07
max  2.140509991e-13

So the diff is still there after my changes, but i still havent confirmed that the input is the same. what is the input? window_df. I.e. the windowed signal_df, right? Lets compare mine to theirs.

Confusion abounds - trying to reshape the main reconstructed signals to match my format, however they are stored as shorter series corresponding only to the skewnorm, or something? I am confused by this. I have no x information, and the sum of lengths is less than half mine.

I think ive found it. The reconstructed signals are not constructed based on the ENTIRE time series, but rather the window to which it belongs. Probably whats introducing the variation.

Q: if they dont 'reconstruct' the interpeak areas, what do they use for the ratio?

NVM! to make things confusing, they calculate the skewnorm for the peak within the window and then again for the whole time series stored in a seperate container `.unmixed_chromatorams`. f me.

the main windowed signal df currently consists of 3 interpeak windows and two peak windows.

my windowed signal df consists of the mixed signal and unmixed signals aligned, with window indexes.

Ok, so.. windowed signal dfs have been proved equal. to be specific, the aligned forms of the mixed and unmixed signals labeled with windows are equal. All information for the main table was derived from the main Chromatogram object so there is no chance of leakage. Thus the inputs are the same, and the variation is occuring from the functions.

Lets now observe the variation in output for my ws_df vs theirs. Even though I beleive I have confirmed that they are the same, I will treat the ws_dfs as different for now. Thus there are 4 combinations:
    - my_function x my_input
    - my_function x main_input
    - main_function x my_input
    - main_function x main_input

Ok ive rewritten my score df factory to be clearer to read and debug. Tomorrow compare your score df output to theirs to see if the variance remains, and then diagnose why it is there.

Ok well, on testing, this is new absolute deviance:

window_idx : 0
time_start : 0.0
time_end : 0.0
signal_area : 5.646230150091469e-10
inferred_area : 1.4605711551318734e-07
signal_variance : 1.8260948444559846e-12
signal_mean : 4.107998663460677e-14
signal_fano_factor : 1.2185851286372618e-12
reconstruction_score : 8.672326745617909e-12

I want to conclude this exploration now, 10E-7 difference of an integration performed on data whose x axis has a precision of two decimal places is not worth it. Curious to know the exact mechanism though.. MOVING ON!

Ok, looks like we're done. the report card is formatted the same as theirs, producing the same values (kind of, havent bothered with rounding formatting). Only thing i havent currently tested is whether the colors work, as they do not appear to show up in pytest environment.

Now we've gotta clean up.

- [ ] remove superf tests
- [ ] fix problems
- [ ] add comments

2024-01-09 09:59:22 im not gna bother cleaning up now, as I essentially need to test the system against my data now to see what needs modification. I will do this in a new project.. I will do this within this project.

2024-01-09 19:53:44

I need to standardise the interfaces of the submodules, moving from loading to baseline correction to window finding etc. Thus they should all take a dataframe 'df' which is expected to contain the necessary columns.
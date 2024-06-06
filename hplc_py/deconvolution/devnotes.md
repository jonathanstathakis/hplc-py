2023-11-28 09:36:45 - Starting hplc-py fork dev. Dev notes here as good as any. Type hinting has become both crucial, and a pain. solutions:
- Use `numpy.typing.NDArray[dtype]` to type hint numpy arrays. note it needs to be numpy types not built-in, i.e. `float` instead of `float`.
- Use pandera `DataFrameModel` and `Series[dtype]` to type hint pandas objects. store the schemas in a seperate file then import for easy reference. This allows you to track the changes

Design Spec:
- establish an explicit core dataframe (database)
- satellite modules use numpy arrays
- Thus Chromatogram and 'fit_peaks' has operates on a df acting as an interface between the df and the satellite modules.

2023-11-30 07:55:08 - update. find_windows is done. took a long time because there was a lot of logic that needed to be parsed (lacking documentation) and tested. I also needed to learn how to use Pandera for Dataframe schema testing, and the importance of type hinting, as well as parameter and return validation. Powerful tools, the knife is sharpened, etc.


2024-01-04 14:30:13 - Ratifying Results. Trying to attain an agreement between my score df and theirs.

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

2023-12-05 12:36:43 - Investigate findwindows settings to produce an expected window range, that is:isolated peaks are covered by the window from left base to right base + buffer, overlapping signals are considered in the same window, again, from left to right base of the *region* + buffer.

2024-01-05 00:50:25 - Solving deviation between results. We're back. My current hypothesis to answer the question of the deviance of the reconstructed signal is the function summing the individual signals - this is only true if there is a summation. Lets detail the current approach: - There is no summation. The current approach calculates the skewnorm distributions for each peak in exactly the same manner as main. It then adds the peak_idx and time_idx. Thats it. So the difference has to be popt. Investigation of the popts has resulted in equal values. Thus somewhere between optimization and reconstruction lies the problem. but there shouldnt be anything in between these two steps.. How is the reconstruction achieved in main, the skewnormas are calculated on a window basis. to recreate the behavior id need to iterate through each window and pass the parameters as a 2d array in the correct order. Not so hard. it already is a 2D array. In their reconstruction, they iterate through a collection of window properties to get the window parameters and time range to compuet the skewnorms, returning an unmixed signal for each window. I can imitate that by calling the windowed signal df and iterating over that. Specifically, they iterate through each peak in each window. Thus i need to iterate not over windows but over peaks. This attempt has failed, reconstructed peaks do not match expectation, hypothesis is that labeling has failed somehow. Right, problem was that I was attempting a many to many join when i thought i was doing a many to one - multiple peak indexes for the same window_idx results in the first peak only being joined. What i need is to label each time range for a given window. but thats already done. what am i doing? x needs to be the window, right? no. time range needs to be the length of the whole mixed signal.. can confirm that my windows match theirs: 3 peaks in the first, 1 in the second. Ok, I have generated a long unmixed peak signal series with peak idx indexing from the main _comptue_skewnorms function. Well there is a surprising result. The two are equal. Where are we? What was the problem? Problem: the AUC of the unmixed signals differed. Possible sources of error: - optimixed parameters: EQUAL - reconstructed signals: EQUAL. So, q: is the input the same? Going back to `fit_assessment` to do some housekeeping. I need to clarify what is going on with my window indexing.

2024-01-07 19:14:28 - Pandas to numpy. window indexing looks fine. back to housekeeping. need to convert all pandas based aggregations to numpy calculations, and preferably perform aggregations seperately for debugging purposes.

2024-01-07 19:47:56 Testing. functions are cleaned up. Now for testing again. The question being worked on earlier was if the input to the score df calculations was different. What is the input? in main, it is.. selecting peak windows then iterating over window indexes.. the area is calculated as.. np.abs(x.values).sum()+1. Note there is no computational difference between np.sum(np.abs(x.values))+1 and np.abs(x.values).sum()+1.

                diff
min -1.156804501e-07
max  2.140509991e-13

                diff
min -1.156804501e-07
max  2.140509991e-13

So the diff is still there after my changes, but i still havent confirmed that the input is the same. what is the input? window_df. I.e. the windowed signal_df, right? Lets compare mine to theirs. Confusion abounds - trying to reshape the main reconstructed signals to match my format, however they are stored as shorter series corresponding only to the skewnorm, or something? I am confused by this. I have no x information, and the sum of lengths is less than half mine. I think ive found it. The reconstructed signals are not constructed based on the ENTIRE time series, but rather the window to which it belongs. Probably whats introducing the variation. Q: if they dont 'reconstruct' the interpeak areas, what do they use for the ratio? NVM! to make things confusing, they calculate the skewnorm for the peak within the window and then again for the whole time series stored in a seperate container `.unmixed_chromatorams`. f me. the main windowed signal df currently consists of 3 interpeak windows and two peak windows. my windowed signal df consists of the mixed signal and unmixed signals aligned, with window indexes. Ok, so.. windowed signal dfs have been proved equal. to be specific, the aligned forms of the mixed and unmixed signals labeled with windows are equal. All information for the main table was derived from the main Chromatogram object so there is no chance of leakage. Thus the inputs are the same, and the variation is occuring from the functions. Lets now observe the variation in output for my ws_df vs theirs. Even though I beleive I have confirmed that they are the same, I will treat the ws_dfs as different for now. Thus there are 4 combinations: my_function x my_input, my_function x main_input, main_function x my_input, main_function x main_input. Ok ive rewritten my score df factory to be clearer to read and debug. Tomorrow compare your score df output to theirs to see if the variance remains, and then diagnose why it is there. Ok well, on testing, this is new absolute deviance:

window_idx : 0
time_start : 0.0
time_end : 0.0
signal_area : 5.646230150091469e-10
inferred_area : 1.4605711551318734e-07
signal_variance : 1.8260948444559846e-12
signal_mean : 4.107998663460677e-14
signal_fano_factor : 1.2185851286372618e-12
reconstruction_score : 8.672326745617909e-12

I want to conclude this exploration now, 10E-7 difference of an integration performed on data whose x axis has a precision of two decimal places is not worth it. Curious to know the exact mechanism though.. MOVING ON! Ok, looks like we're done. the report card is formatted the same as theirs, producing the same values (kind of, havent bothered with rounding formatting). Only thing i havent currently tested is whether the colors work, as they do not appear to show up in pytest environment.

2024-01-09 19:53:44 - Refactoring. I need to standardise the interfaces of the submodules, moving from loading to baseline correction to window finding etc. Thus they should all take a dataframe 'df' which is expected to contain the necessary columns.

2024-03-04 11:53:43 Debugging deconvolution for the 15th time. Comparing the inputs and outputs of popt reveals no significant differences bar the upper bound scale of peak 3, which is ~59 to clabs 7. No clear reason why so will need to investigate that first.

2024-03-04 12:15:32 - The scale ub calculation was wrong, used the division of the window end time value rather than the division of the length of each window. fixed, but no change to popt differences. Next point to check will be the deconvolution input. The input time series has been a problem before, could be again. Looks right to me.. can we test it with the cremerlab params? Fixed it. Was trying to deconvolve based on X_idx units rather than X.  The internal references to definitions makes it hard to debug due to loss of encapsulation.

2024-03-06 11:53:00 - Trying to design the progress bar. What we currently have is an algorithm that takes a max_nfev and divides it into subsets. This therefore imposes the requirement that each subset gets enough nfev, which is an error we have to be able to handle. This also means that the optimization keeps running until max_nfev is reached, rather than terminating as expected when a solution is found. So thats two things to do: handle the case of an error during optimization, tell it to stop when optimization is reached through successful termination condition. All of this was based on the idea that I could collect intermediate results. I dont think thats true. if it errors out when it reaches input max_nfev, how can I collect the results? What does scipy curve_fit actually do? because it looks like it masks a lot of information returned by least_squares. Ok so passing verbose to least_squares will allow us to print a progress report during iterations. just add the kwargs options, least squares has a 'max_nfev' option that should do what we expect. It does, without error. Is curve fit throwing the error? yes it is. why the fuck did they write it like that. What if we recreate curve fit without it.. So the error occurs in all cases when least_squares result attribute `success` is False. no way around that. again, what does curve fit provide over a direct least_squares call? See [scipy_curve_fit_outline](./scipy_curve_fit_outline.md) for the outline. Its minimal, a bunch of input validation and edge case handling which we could reimplement without that rediculous error raising.. Considering that I dont care about any of it and my input is already validated i can just drop curve fit in favor of least squares.. Now we've got the elapsed times, and we can get all the information from least_squares while running partial fits. To implement the progress bar we need to again base it off of the max_nfev submitted, and implement a loop that continues until the success message is passed. What does that mean? what that means is that every window is successfully fitted. How do we report that? each window is fitted the same number of times, treated independently. so one progress bar for windows, another for the intermediate fits. Lets be explicit about the call heirarchy in deconvolution: 1. deconvolution_pipeline, 2. curve_fit_windows, 3. popt_factory 4. get_popt - iterates over the max_nfev divisions, 5. curve_fit_, 6. least_squares

2024-03-07 00:19:21 -  Time to assemble my analysis tool. At a basic level it just needs to provide the fit report.

2024-04-24 23:00:11 I need to acknowledge (which I have before) the importance of seperating my stream-of-conciousness working from the chapters. For example, my work over the last two weeks has been contained in what is know prefixed with "logbook_". These are at times random rambling around a topic or problem trying various methods to move foreward. Thanks to the capabilities of digital document processing, it is easy enough, and natural, to attempt to take those documents and edit them in place to produce a polished copy ready for publication. Unfortunately that results in both a loss of recall when I wish to go back to my notes, and is dishonest in presentation of process. thus i will prefix the logbook with 'logbook_', and chaper with "ch_{chapter_name}_{section_name}"
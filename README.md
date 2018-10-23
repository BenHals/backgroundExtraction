Automatically extract background and foreground components of an image.

images to test should be placed in /input

The main starting point of the program is process.py, calling "python process.py --file FILENAME" will run all steps of the process automatically, opeining a unity game process when finished.
All output is stored in a created folder in /output.

seg.py runs the component extraction process. 
	--blocksize sets the size of segmentation windows
	--nulti sets the threshold multiplier parameter
	--texthresh sets the texture threshold for window comparison
	--colthresh sets the color threshold for window comparison
	--file sets the input file
	
	These all have been set to reasonable defaults, so only need to be changed if an input image is not extracting well
	
createbg.py synthesizes a background

genunity.py generates the unity scene, builds it and starts the unity environment
	--unity sets the unity exe location
	
	This must be set to where unity is installed on your computer!
	
	The unity scene file and exe are stored in the output folder in /output, the scene file can be opened in unity and the exe can be run for the interactive game. 


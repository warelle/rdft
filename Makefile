all: main

main: main.py lu.py rdft.py
	python main.py

iteration: iteration.py lu.py rdft.py
	python iteration.py

clean:
	rm lu.pyc rdft.pyc

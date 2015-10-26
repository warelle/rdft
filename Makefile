all: main

main: main.py lu.py rdft.py
	python main.py

wrapper: wrapper.py iteration.py lu.py rdft.py
	python wrapper.py

clean:
	rm *.pyc

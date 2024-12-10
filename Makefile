.PHONY: read_weight compile


read_weight: read_weight.py
	"C:/Program Download/Python-complier/python.exe" $<

compile: main_model_learn.py
	"C:/Program Download/Python-complier/python.exe" $<
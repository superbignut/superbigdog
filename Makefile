.PHONY: read_weight compile rp


read_weight: read_weight.py
	"C:/Program Download/Python-complier/python.exe" $<

compile: main_model_learn.py
	"C:/Program Download/Python-complier/python.exe" $<


rp: read_and_plot.py
	"C:/Program Download/Python-complier/python.exe" $<
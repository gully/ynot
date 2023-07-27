all: ms.pdf

%.pdf: %.tex
	tectonic ms.tex -o final
	rm -rf *Notes.bib
	rm -rf *.log
	rm -rf *.bbl
	rm -rf *.aux
	rm -rf *.out
	rm -rf *.blg
	rm -rf *.dvi
	open final/ms.pdf

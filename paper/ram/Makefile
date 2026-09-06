.PHONY: all paper supplementary clean

TEX = pdflatex -interaction=nonstopmode

paper: main.pdf

main.pdf: main.tex sections/*.tex bibliography.bib IEEEtran.cls IEEEtran.bst figures/*
	$(TEX) '-jobname=main' main.tex
	bibtex main
	$(TEX) '-jobname=main' main.tex
	$(TEX) '-jobname=main' main.tex

supplementary: supplementary/supplementary.pdf

supplementary/supplementary.pdf: supplementary/supplementary.tex supplementary/sections/*.tex
	cd supplementary && $(TEX) '-jobname=supplementary' supplementary.tex
	cd supplementary && bibtex supplementary
	cd supplementary && $(TEX) '-jobname=supplementary' supplementary.tex
	cd supplementary && $(TEX) '-jobname=supplementary' supplementary.tex

all: paper supplementary

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
	rm -f supplementary/*.aux supplementary/*.bbl supplementary/*.blg supplementary/*.log supplementary/*.out supplementary/*.toc

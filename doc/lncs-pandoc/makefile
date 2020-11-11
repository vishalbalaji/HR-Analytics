#.DEFAULT_GOAL := all
PAPER += paper

all:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ./tmp.tex -C --template=llncs.latex $(PAPER).md
	sed --regexp-extended 's/tbl\./Table/g; s/fig\./Fig\./g; s/ref-/ref_/g; s/\{\[\}\\protect\\hyperlink/\\cite/g; s/\{.\}\{\]\}//g; s/\\begin\{CSLReferences\}\{0\}\{0\}/\\begin\{thebibliography\}\{8\}/;s/\\end\{CSLReferences\}/\\end\{thebibliography\}/; s/\\leavevmode\\hypertarget/\\bibitem/; s/\\CSLRightInline//; s/\{\}\%//' tmp.tex | awk '!/\\hypertarget\{references\}.*|\\section\*\{References\}\\label\{references\}\}|\\addcontentsline\{toc\}\{section\}\{References\}|\\CSLLeftMargin.*|\\hypertarget\{refs\}\{\}/' > tmp1.tex
	mv tmp1.tex tmp.tex
	pdflatex -quiet tmp.tex
	pdflatex -quiet tmp.tex
	cp tmp.pdf ../final/$(PAPER).pdf
	cp tmp.tex ../final/$(PAPER).tex
	rm -r tmp.* $(PAPER)* *.bib imgs

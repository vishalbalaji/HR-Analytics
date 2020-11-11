#.DEFAULT_GOAL := all
PAPER += paper

all:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).pdf -C --template=llncs.latex $(PAPER).md
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).tex -C --template=llncs.latex $(PAPER).md
	rm -f $(PAPER).{log,out,aux,bbl,blg}
	rm -r $(PAPER).md *.bib imgs
tex:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ./tmp.tex -C --template=llncs_no_csl.latex $(PAPER).md
	sed --regexp-extended 's/\{\[\}\\protect\\hyperlink/\\cite/; s/\{\]\}//g; s/\\begin\{CSLReferences\}\{0\}\{0\}/\\begin\{thebibliography\}\{8\}/;s/\\end\{CSLReferences\}/\\end\{thebibliography\}/; s/\\leavevmode\\hypertarget/\\bibitem/; s/\\CSLRightInline//' tmp.tex | awk '!/\\hypertarget\{references\}.*|\\section\*\{References\}\\label\{references\}\}|\\addcontentsline\{toc\}\{section\}\{References\}|\\CSLLeftMargin.*|\\hypertarget\{refs\}\{\}/' > ../$(PAPER).tex
	rm -f $(PAPER).{log,out,aux,bbl,blg}
	rm -r $(PAPER).md *.bib tmp.tex imgs
pdf:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).pdf -C --template=llncs.latex $(PAPER).md
	rm -f $(PAPER).{log,out,aux,bbl,blg}
	rm -r $(PAPER).md *.bib imgs

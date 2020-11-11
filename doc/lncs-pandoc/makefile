#.DEFAULT_GOAL := all
PAPER += paper

all:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).pdf -C --template=llncs.latex $(PAPER).md
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).tex -C --template=llncs.latex $(PAPER).md
	rm -rf $(PAPER).{log,out,aux,bbl,blg,md} *.bib imgs
pdf:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ../$(PAPER).pdf -C --template=llncs.latex $(PAPER).md
	rm -rf $(PAPER).{log,out,aux,bbl,blg,md} *.bib imgs
bibpdf:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ./tmp.tex -C --template=llncs_no_csl.latex $(PAPER).md
	sed --regexp-extended 's/\{\[\}\\protect\\hyperlink/\\cite/; s/\{\]\}//g; s/\\begin\{CSLReferences\}\{0\}\{0\}/\\begin\{thebibliography\}\{8\}/;s/\\end\{CSLReferences\}/\\end\{thebibliography\}/; s/\\leavevmode\\hypertarget/\\bibitem/; s/\\CSLRightInline//' tmp.tex | awk '!/\\hypertarget\{references\}.*|\\section\*\{References\}\\label\{references\}\}|\\addcontentsline\{toc\}\{section\}\{References\}|\\CSLLeftMargin.*|\\hypertarget\{refs\}\{\}/' > tmp1.tex
	mv tmp1.tex tmp.tex
	pdflatex tmp.tex
	cp tmp.pdf ../$(PAPER).pdf
	rm -r tmp.* $(PAPER)* *.bib imgs
tex:
	cp -r ../$(PAPER).md ../refs/refs.bib ../imgs .
	pandoc --filter pandoc-crossref --from markdown+auto_identifiers -o ./tmp.tex -C --template=llncs_no_csl.latex $(PAPER).md
	sed --regexp-extended 's/\{\[\}\\protect\\hyperlink/\\cite/; s/\{\]\}//g; s/\\begin\{CSLReferences\}\{0\}\{0\}/\\begin\{thebibliography\}\{8\}/;s/\\end\{CSLReferences\}/\\end\{thebibliography\}/; s/\\leavevmode\\hypertarget/\\bibitem/; s/\\CSLRightInline//' tmp.tex | awk '!/\\hypertarget\{references\}.*|\\section\*\{References\}\\label\{references\}\}|\\addcontentsline\{toc\}\{section\}\{References\}|\\CSLLeftMargin.*|\\hypertarget\{refs\}\{\}/' > ../$(PAPER).tex
	rm -rf $(PAPER).{log,out,aux,bbl,blg,md} *.tex *.bib imgs

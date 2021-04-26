# 
# @author: AMOUSSOU Z. Kenneth
# @date: 28.11.2020
#
SOURCE_FILE=report
FILTERS=


help:
	@echo "Generate the report"
	@echo "   build: generate the output pdf file"
	@echo "   clean: delete all generated output files"
	@echo

build:
	@xelatex ${SOURCE_FILE}.tex
	@bibtex ${SOURCE_FILE}.aux > /dev/null
	@xelatex ${SOURCE_FILE} > /dev/null

clean:
	@echo "cleaning ..."
	@find . -type f \( -name "*.log" -o -name "report.pdf" -o -name "*.bbl" -o -name \
										"*.blg" -o -name "*.aux" -o -name "*.out" -o -name "*.xml" \
										-o -name "*.toc" \) -exec rm {} \;


#!/bin/sh
# Рендерить діаграми UML (PlantUML) і схему ієрархії моделей (graphviz) у папку outputs.
set -e
cd "$(dirname "$0")"
plantuml -tpng -charset UTF-8 -o ../outputs ./*.puml
dot -Tpng -o ../outputs/hierarchy.png hierarchy.dot

#!/bin/sh
# Рендерить діаграми UML (PlantUML) і діаграму класів, отриману зворотним проектуванням коду (pyreverse).
set -e
cd "$(dirname "$0")"
plantuml -tpng -charset UTF-8 -o ../outputs ./*.puml

uv run pyreverse -o dot -k -p asm -d . ../kg_asm.py ../cli.py >/dev/null
sed -e 's/fontcolor="green"/fontcolor="black"/g' \
    -e 's/^rankdir=BT$/rankdir=BT\ngraph [fontname="Arial", dpi=200, nodesep=0.35, ranksep=0.45, pack=true, packmode="array_u3"];\nnode [fontname="Arial", fontsize=11];\nedge [fontname="Arial", fontsize=10];/' \
    classes_asm.dot > reverse_classes.dot
rm -f classes_asm.dot packages_asm.dot
dot -Tpng -o ../outputs/reverse_classes.png reverse_classes.dot

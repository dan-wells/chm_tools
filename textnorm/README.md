# OpenGRM Thrax grammars for text normalization

Compile grammars:
```
thraxcompiler --input_grammar=byte.grm --output_far=byte.far
thraxcompiler --input_grammar=textnorm.grm --output_far=textnorm.far
```

Interactive rewrites:
```
thraxrewrite-tester --far=textnorm.far --rules=LENZ_MAP
```

Batch rewrites (output needs cleaning up):
```
cat orig.txt | thraxrewrite-tester --far=textnorm.far --rules=LENZ_MAP > norm.txt
```

---
Brian Roark, Richard Sproat, Cyril Allauzen, Michael Riley, Jeffrey Sorensen and Terry Tai. 2012. _The OpenGrm open-source finite-state grammar software libraries_. In Proceedings of the ACL 2012 System Demonstrations, pp. 61-66. ``http://www.opengrm.org``. 

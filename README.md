# chm_tools

Collection of tools for working with the [Corpus of Historical Mapudungun](https://benmolineaux.github.io/).

Original data files are in XML format, using the [Text Encoding Initiative](https://tei-c.org/ns/1.0/) namespace.
Parsed data files are output in [CoNNL-U format](https://universaldependencies.org/docs/format.html),
allowing for differences in tag sets used: POS tags are in the XPOSTAG column, i.e. 'language-specific'.
A many-to-one mapping into the UPOSTAG set may be done later.

from embedder import embedder

english = embedder("Blockchain enables decentralized applications.")
hindi = embedder("ब्लॉकचेन विकेंद्रीकृत एप्लिकेशन को सक्षम बनाता है।")
multiple = embedder(["Hello world", "नमस्ते दुनिया"])

print(english)
print(english.shape)   # (1024,)
print(hindi.shape)     # (1024,)
print(multiple.shape)  # (2, 1024)  
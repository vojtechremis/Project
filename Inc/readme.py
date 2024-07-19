def create(content, filename = 'README.md'):
    with open(filename, 'w') as readme_file:
        readme_file.write(content)

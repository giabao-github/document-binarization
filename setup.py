from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
	requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
	name="document-binarization",
	version="0.1.0",
	author="CV Team",
	description="Document image binarization and text enhancement for OCR",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Developers",
		"Topic :: Scientific/Engineering :: Image Processing",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
	],
	python_requires=">=3.8",
	install_requires=requirements,
	entry_points={
		"console_scripts": [
			"binarize=scripts.binarize:main",
			"evaluate-binarization=scripts.evaluate:main",
			"optimize-params=scripts.optimize:main",
		],
	},
)
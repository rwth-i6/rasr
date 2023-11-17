
#include <util/utillib.h>
#include <stdio.h>

// provide dummy implementation for unresolved symbols

int file_readable(char *fname) {
	FILE* f = fopen(fname, "r");
	if(f) {
		fclose(f);
		return 1;
	}
	return 0;
}


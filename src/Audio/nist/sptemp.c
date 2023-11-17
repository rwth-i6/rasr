
/* LINTLIBRARY */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SPHERE_LIBRARY_CODE
#include <sp/sphere.h>

char *sptemp() {
    static char *template = NULL;
    char *result;
    int fd;
    if (template == NULL) {
	const char *tmpdir;
	tmpdir = getenv("TMPDIR");
	if (tmpdir == NULL) tmpdir = "/tmp";
	template = malloc(strlen(tmpdir) + 1 + strlen(TEMP_BASE_NAME) + 6 + 1);
	strcpy(template, tmpdir);
	strcat(template, "/");
	strcat(template, TEMP_BASE_NAME);
	strcat(template, "XXXXXX");
    }

    result = strdup(template);
    fd = mkstemp(result);
    if (fd < 0) {
	free(result);
	return NULL;
    }
    close(fd);
    return result;
}

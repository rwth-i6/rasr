/******************************************************************************
*                                                                             *
*       Copyright (C) 1992-1995 Tony Robinson                                 *
*                                                                             *
*       See the file LICENSE for conditions on distribution and usage         *
*                                                                             *
******************************************************************************/

# include <math.h>
# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <setjmp.h>
# include <unistd.h>
#ifdef MSDOS
# include <io.h>
# include <fcntl.h>
#endif
/* Changed by Jon Fiscus */
# include <util/fob.h>
# include <sp/shorten/shorten.h>

#ifdef fread
#       undef fread
#endif
#define fread(a,b,c,d)          fob_fread((a),(b),(c),(d))
 
#ifdef fwrite
#       undef fwrite
#endif
#define fwrite(a,b,c,d)         fob_fwrite((a),(b),(c),(d))
 
#ifdef putc
#       undef putc
#endif
#define putc(a,b)               fob_putc((a),(b))

#ifdef getc
#       undef getc
#endif
#define getc(a)                 fob_getc((a))

#ifdef fclose
#       undef fclose
#endif
#define fclose(a)               fob_fclose ((a))

#ifdef fopen
#       undef fopen
#endif
#define fopen(a,b)              fob_fopen((a),(b))

extern void license();
extern int wav2lpc_init();

#ifndef MSDOS
char *readmode = "r";
char *writemode = "w";
# define FILESUFFIX ".shn"
#else
char *readmode = "rb";
char *writemode = "wb";
#endif

char	*argv0 = "shorten";
int	getc_exit_val;
char	*filenameo = NULL;
FOB	*fileo = NULL;

# define V2LPCQOFFSET (1 << LPCQUANT);

# define UINT_PUT(val, nbit, file) \
  if(version == 0) uvar_put((unsigned long) val, nbit, file); \
  else ulong_put((unsigned long) val, file)

# define UINT_GET(nbit, file) \
  ((version == 0) ? uvar_get(nbit, file) : ulong_get(file))

# define VAR_PUT(val, nbit, file) \
  if(version == 0) var_put((unsigned long) val, nbit - 1, file); \
  else var_put((unsigned long) val, nbit, file)

void init_offset(offset, nchan, nblock, ftype) long **offset; int nchan,
     nblock, ftype; {
  long mean = 0;
  int  chan, i;

  /* initialise offset */
  switch(ftype) {
  case TYPE_AU1:
  case TYPE_S8:
  case TYPE_S16HL:
  case TYPE_S16LH:
  case TYPE_ULAW:
  case TYPE_AU2:
    mean = 0;
    break;
  case TYPE_U8:
    mean = 0x80;
    break;
  case TYPE_U16HL:
  case TYPE_U16LH:
    mean = 0x8000;
    break;
  default:
    update_exit(1, "unknown file type: %d\n", ftype);
  }

  for(chan = 0; chan < nchan; chan++)
    for(i = 0; i < nblock; i++)
      offset[chan][i] = mean;
}

float Satof(string) char *string; {
  int i, rval = 1;

  /* this should have tighter checking */
  for(i = 0; i < strlen(string) && rval == 1; i++)
    if(string[i] != '.' && (string[i] < '0' || string[i] > '9'))
      usage_exit(1, "non-parseable float: %s\n", string);
  return(atof(string));
}

float* parseList(maxresnstr, nchan) char *maxresnstr; int nchan; {
  int   nval;
  char  *str, *floatstr;
  float *floatval;

  /* copy maxresnstr to temporary space, str */
  str = malloc(strlen(maxresnstr) + 1);
  strcpy(str, maxresnstr);

  /* grab space for the floating point parses */
  floatval = pmalloc((ulong) (nchan * sizeof(*floatval)));

  /* loop for all floats in the arguement */
  floatstr = strtok(str, ",");
  floatval[0] = Satof(floatstr);

  for(nval = 1; (floatstr = strtok(NULL, ",")) != NULL && nval < nchan;nval++)
    floatval[nval] = Satof(floatstr);

  for(; nval < nchan; nval++)
    floatval[nval] = floatval[nval - 1];

  free(str);

  return(floatval);
}


int shorten(stdi, stdo, argc, argv) FOB *stdi, *stdo; int argc; char **argv; {
  long 	**buffer, *buffer1, **offset;
  long	lpcqoffset = 0;
  int   version = FORMAT_VERSION, extract = 0, lastbitshift = 0, bitshift = 0;
  int   hiloint = 1, hilo = !(*((char*) &hiloint));
  int   ftype = hilo ? TYPE_S16HL : TYPE_S16LH;
  char  *magic = MAGIC, *filenamei = NULL;
  char	*tmpfilename = NULL;
  char  *maxresnstr = DEFAULT_MAXRESNSTR;
  FOB  *filei;
  int	blocksize = DEFAULT_BLOCK_SIZE, nchan = DEFAULT_NCHAN;
  int	i, chan, nwrap, nskip = DEFAULT_NSKIP, ndiscard = DEFAULT_NDISCARD;
  int	*qlpc = NULL, maxnlpc = DEFAULT_MAXNLPC, nmean = UNDEFINED_UINT;
  int   quanterror = DEFAULT_QUANTERROR, minsnr = DEFAULT_MINSNR, nfilename;
  int	ulawZeroMerge = 0;
  char	*minusstr  = "-";
  extern char *hs_optarg;
  extern int   hs_optind;

  /* this block just processes the command line arguments */
  { int c;

    hs_resetopt();
    while((c = hs_getopt(argc, argv, "a:b:c:d:hlm:n:p:q:r:t:uv:x")) != -1)
      switch(c) {
      case 'a':
	if((nskip = atoi(hs_optarg)) < 0)
	  usage_exit(1, "number of bytes to copy must be positive\n");
	break;
      case 'b':
	if((blocksize = atoi(hs_optarg)) <= 0)
	  usage_exit(1, "block size must be greater than zero\n");
	break;
      case 'c':
	if((nchan = atoi(hs_optarg)) <= 0)
	  usage_exit(1, "number of channels must be greater than zero\n");
	break;
      case 'd':
	if((ndiscard = atoi(hs_optarg)) < 0)
	  usage_exit(1, "number of bytes to discard must be positive\n");
	break;
      case 'h':
	printf("%s: version %d.%s: (c) 1992-1995 Tony Robinson\n", argv0,
      	       FORMAT_VERSION, BUGFIX_RELEASE);
#ifdef OLDHELP
	printf("usage: %s [-hx] [-a #byte] [-b #sample] [-c #channel] [-d #discard]\n\t[-m #block] [-p #delay] [-q #bit] [-r #bit] [-t filetype]\n\t[input file] [output file]\n", argv0);
#endif
	printf("%s: usage: %s {options} [input file] [output file]\n", argv0,
               argv0);
	printf("\t-a %d\tbytes to copy verbatim to align file\n",
	       DEFAULT_NSKIP);
	printf("\t-b %d\tblock size\n", DEFAULT_BLOCK_SIZE); 
	printf("\t-c %d\tnumber of channels\n", DEFAULT_NCHAN); 
	printf("\t-d %d\tbytes to discard before compression or decompression\n", DEFAULT_NDISCARD); 
	printf("\t-h\thelp (this message)\n");
	printf("\t-l\tprint the license giving the distribution and usage conditions\n");
	printf("\t-m %d\tnumber of past block for mean estimation\n",
                (FORMAT_VERSION < 2) ? DEFAULT_V0NMEAN : DEFAULT_V2NMEAN);
	printf("\t-n %d\tminimum signal to noise ratio in dB (%d == lossless coding)\n",
      	       DEFAULT_MINSNR, DEFAULT_MINSNR);
	printf("\t-p %d\tmaximum LPC predictor order (0 == fast polynomial predictor)\n", DEFAULT_MAXNLPC);
	printf("\t-q %d\tacceptable quantisation error in bits\n",
	       DEFAULT_QUANTERROR);
	printf("\t-r %s\tmaximum number of bits per sample (%s == lossless coding)\n",
	       DEFAULT_MAXRESNSTR, DEFAULT_MAXRESNSTR);
	printf("\t-t s16\tfiletype {ulaw,s8,u8,s16,u16,s16x,u16x,s16hl,u16hl,s16lh,u16lh}\n");
	printf("\t-u\tmerge the two zero codes in ulaw files\n");
	printf("\t-v %d\tformat version number\n", FORMAT_VERSION);
	printf("\t-x\textract (all other options except -a and -d are ignored)\n");

	basic_exit(0);
	break;
      case 'l':
	license();
	basic_exit(0);
	break;
      case 'm':
	if((nmean = atoi(hs_optarg)) < 0)
	  usage_exit(1,	"number of blocks for mean estimation must be positive\n");
	break;
      case 'n':
	if((minsnr = atoi(hs_optarg)) < 0)
	  usage_exit(1, "Useful signal to noise ratios are positive\n");
	break;
      case 'p':
	maxnlpc = atoi(hs_optarg);
	if(maxnlpc < 0 || maxnlpc > MAX_LPC_ORDER)
	  usage_exit(1, "linear prediction order must be in the range 0 ... %d\n", MAX_LPC_ORDER);
	break;
      case 'q':
	if((quanterror = atoi(hs_optarg)) < 0)
	  usage_exit(1, "quantisation level must be positive\n");
	break;
      case 'r':
	maxresnstr = hs_optarg;
	break;
      case 't':
	if     (!strcmp(hs_optarg, "au"))   ftype = TYPE_GENERIC_ULAW;
	else if(!strcmp(hs_optarg, "ulaw")) ftype = TYPE_GENERIC_ULAW;
	else if(!strcmp(hs_optarg, "s8"))   ftype = TYPE_S8;
	else if(!strcmp(hs_optarg, "u8"))   ftype = TYPE_U8;
	else if(!strcmp(hs_optarg, "s16"))  ftype = hilo?TYPE_S16HL:TYPE_S16LH;
	else if(!strcmp(hs_optarg, "u16"))  ftype = hilo?TYPE_U16HL:TYPE_U16LH;
	else if(!strcmp(hs_optarg, "s16x")) ftype = hilo?TYPE_S16LH:TYPE_S16HL;
	else if(!strcmp(hs_optarg, "u16x")) ftype = hilo?TYPE_U16LH:TYPE_U16HL;
	else if(!strcmp(hs_optarg, "s16hl"))ftype = TYPE_S16HL;
	else if(!strcmp(hs_optarg, "u16hl"))ftype = TYPE_U16HL;
	else if(!strcmp(hs_optarg, "s16lh"))ftype = TYPE_S16LH;
	else if(!strcmp(hs_optarg, "u16lh"))ftype = TYPE_U16LH;
	else usage_exit(1, "unknown file type: %s\n", hs_optarg);
	break;
      case 'u':
	ulawZeroMerge = 1;
	break;
      case 'v':
	version = atoi(hs_optarg);
	if(version < 0 || version > MAX_SUPPORTED_VERSION)
	  usage_exit(1,
  	    "currently supported versions are in the range %d ... %d\n",
      	    MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION);
	break;
      case 'x':
	extract = 1;
	break;
      default:
	usage_exit(1, NULL);
	break;
      }
  }

  wav2lpc_init();

  if(nmean == UNDEFINED_UINT)
    nmean = (version < 2) ? DEFAULT_V0NMEAN : DEFAULT_V2NMEAN;

  if(blocksize <= NWRAP)
    usage_exit(1, "blocksize must be greater than %d\n", NWRAP);

  if(maxnlpc >= blocksize)
    usage_exit(1, "the predictor order must be less than the block size\n");

  if(ulawZeroMerge == 1 && ftype != TYPE_GENERIC_ULAW)
    usage_exit(1, "the -u flag is only applicable to otherwise lossless ulaw coding\n");

  if(ftype == TYPE_GENERIC_ULAW) {
    if((Satof(maxresnstr) != Satof(DEFAULT_MAXRESNSTR)
	|| quanterror != DEFAULT_QUANTERROR))
      ftype = TYPE_ULAW;
    else if(version < 2 || ulawZeroMerge == 1)
      ftype = TYPE_AU1;
    else
      ftype = TYPE_AU2;
  }

  /* mean compensation is not supported for TYPE_AU1 or TYPE_AU2 */
  if(ftype == TYPE_AU1 || ftype == TYPE_AU2)
    nmean = 0;

  /* this chunk just sets up the input and output files */
  nfilename = argc - hs_optind;
  switch(nfilename) {
  case 0:
    filenamei = minusstr;
    filenameo = minusstr;
    break;
  case 1: {
#ifndef MSDOS
    int oldfilelen, suffixlen, maxlen;

    filenamei  = argv[argc - 1];
    oldfilelen = strlen(filenamei);
    suffixlen  = strlen(FILESUFFIX);
    maxlen     = oldfilelen + suffixlen;
    tmpfilename = pmalloc((ulong) (maxlen + 1));
    strcpy(tmpfilename, filenamei);

    if(extract) {
      int newfilelen = oldfilelen - suffixlen;
      if(strcmp(filenamei + newfilelen, FILESUFFIX))
	usage_exit(1,"file name does not end in %s: %s\n", FILESUFFIX,
		   filenamei);
      tmpfilename[newfilelen] = '\0';
    }
    else
      strcat(tmpfilename, FILESUFFIX);

    filenameo = tmpfilename;
#else
    usage_exit(1, "must specify both input and output file when running under DOS\n");
#endif
    break;
  }
  case 2:
    filenamei = argv[argc - 2];
    filenameo = argv[argc - 1];
    break;
  default:
    usage_exit(1, NULL);
  }

  if(strcmp(filenamei, minusstr)) {
      if((filei = fopen(filenamei, readmode)) == NULL)
      perror_exit("fopen(\"%s\", \"%s\")", filenamei, readmode);
  }
  else {
    filei = stdi;
#ifdef MSDOS
    setmode(fileno(filei), O_BINARY);
#endif
  }

  if(strcmp(filenameo, minusstr)) {
    if((fileo = fopen(filenameo, writemode)) == NULL)
      perror_exit("fopen(\"%s\", \"%s\")", filenameo, writemode);
  }
  else {
    fileo = stdo;
#ifdef MSDOS
    setmode(fileno(fileo), O_BINARY);
#endif
  }

  /* discard header on input file - can't rely on fseek() here */
  if(ndiscard != 0) {
    char discardbuf[BUFSIZ];

    for(i = 0; i < ndiscard / BUFSIZ; i++)
      if(fread(discardbuf, BUFSIZ, 1, filei) != 1)
	usage_exit(1, "EOF on input when discarding header\n");

    if(ndiscard % BUFSIZ != 0)
      if(fread(discardbuf, ndiscard % BUFSIZ, 1, filei) != 1)
	usage_exit(1, "EOF on input when discarding header\n");
  }

  if(!extract) {
    float *maxresn = (float  *)0;
    int   nread, nscan = 0, vbyte = MAX_VERSION + 1;

    nwrap = MAX(NWRAP, maxnlpc);

    /* grab some space for the input buffers */
    buffer  = long2d((ulong) nchan, (ulong) (blocksize + nwrap));
    buffer1 = pmalloc((ulong) (blocksize * sizeof(*buffer1)));
    offset  = long2d((ulong) nchan, (ulong) MAX(1, nmean));

    maxresn = parseList(maxresnstr, nchan);
    for(chan = 0; chan < nchan; chan++)
      if(maxresn[chan] < MINBITRATE)
    	usage_exit(1,"channel %d: expected bit rate must be >= %3.1f: %3.1f\n",
		   MINBITRATE, chan, maxresn[chan]);
      else
	maxresn[chan] -= 3.0;

    for(chan = 0; chan < nchan; chan++) {
      for(i = 0; i < nwrap; i++) buffer[chan][i] = 0;
      buffer[chan] += nwrap;
    }

    if(maxnlpc > 0)
      qlpc = (int*) pmalloc((ulong) (maxnlpc * sizeof(*qlpc)));

    init_offset(offset, nchan, MAX(1, nmean), ftype);

    /* verbatim copy of skip bytes from input to output checking for the
       existence of magic number in header, and defaulting to internal storage
       if that happens */
    if(version >= 2) {
      while(nskip - nscan > 0 && vbyte > MAX_VERSION) {
	int byte = getc_exit(filei);
	if(magic[nscan] != '\0' && byte == magic[nscan]) nscan++;
	else if(magic[nscan] == '\0' && byte <= MAX_VERSION) vbyte = byte;
	else {
	  for(i = 0; i < nscan; i++)
	    putc_exit(magic[i], fileo);
	  if(byte == magic[0]) {
	    nskip -= nscan;
	    nscan = 1;
	  }
	  else {
	    putc_exit(byte, fileo);
	    nskip -= nscan + 1;
	    nscan = 0;
	  }
	}
      }
      if(vbyte > MAX_VERSION) {
	for(i = 0; i < nscan; i++)
	  putc_exit(magic[i], fileo);
	nskip -= nscan;
	nscan = 0;
      }
    }

    /* write magic number */
    if(fwrite(magic, strlen(magic), 1, fileo) != 1)
      usage_exit(1, "could not write the magic number\n");

    /* write version number */
    putc_exit(version, fileo);

    /* initialise the fixed length file read for the uncompressed stream */
    fread_type_init();

    /* initialise the variable length file write for the compressed stream */
    var_put_init();

    /* put file type and number of channels */
    UINT_PUT(ftype, TYPESIZE, fileo);
    UINT_PUT(nchan, CHANSIZE, fileo);

    /* put blocksize if version > 0 */
    if(version == 0) {
      if(blocksize != DEFAULT_BLOCK_SIZE) {
	uvar_put((ulong) FN_BLOCKSIZE, FNSIZE, fileo);
	UINT_PUT(blocksize, (int) (log((double) DEFAULT_BLOCK_SIZE) / M_LN2),
		 fileo);
      }
    }
    else {
      UINT_PUT(blocksize, (int) (log((double) DEFAULT_BLOCK_SIZE) / M_LN2),
	       fileo);
      UINT_PUT(maxnlpc, LPCQSIZE, fileo);
      UINT_PUT(nmean, 0, fileo);
      UINT_PUT(nskip, NSKIPSIZE, fileo);
      if(version == 1) {
	for(i = 0; i < nskip; i++) {
	  int byte = getc_exit(filei);
	  uvar_put((ulong) byte, XBYTESIZE, fileo);
	}
      }
      else {
	if(vbyte <= MAX_VERSION) {
	  for(i = 0; i < nscan; i++)
	    uvar_put((ulong) magic[i], XBYTESIZE, fileo);
	  uvar_put((ulong) vbyte, XBYTESIZE, fileo);
	}
	for(i = 0; i < nskip - nscan - 1; i++) {
	  int byte = getc_exit(filei);
	  uvar_put((ulong) byte, XBYTESIZE, fileo);
	}
	lpcqoffset = V2LPCQOFFSET;
      }
    }

    /* this is the main read/code/write loop for the whole file */
    while((nread = fread_type(buffer, ftype, nchan, blocksize, filei)) != 0) {

      /* put blocksize if changed */
      if(nread != blocksize) {
	uvar_put((ulong) FN_BLOCKSIZE, FNSIZE, fileo);
	UINT_PUT(nread, (int) (log((double) blocksize) / M_LN2), fileo);
	blocksize = nread;
      }

      /* loop over all channels, processing each channel in turn */
      for(chan = 0; chan < nchan; chan++) {
	float sigbit;  /* PT expected root mean squared value of the signal */
      	float resbit;  /* PT expected root mean squared value of the residual*/
      	long coffset, *cbuffer = buffer[chan], fulloffset = 0;
	int  fnd = 0, resn = 0, nlpc = 0;

	/* force the lower quanterror bits to be zero */
	if(quanterror != 0) {
	  long offset = (1L << (quanterror - 1));
	  for(i = 0; i < blocksize; i++)
	    cbuffer[i] = (cbuffer[i] + offset) >> quanterror;
	}

	/* merge both ulaw zeros if required */
	if(ulawZeroMerge == 1)
	  for(i = 0; i < blocksize; i++)
	    if(cbuffer[i] == NEGATIVE_ULAW_ZERO) cbuffer[i]=POSITIVE_ULAW_ZERO;

	/* test for excessive and exploitable quantisation, and exploit!! */
	bitshift = find_bitshift(cbuffer, blocksize, ftype) + quanterror;
	if(bitshift > NBITPERLONG) bitshift = NBITPERLONG;

      	/* find mean offset : N.B. this code duplicated */
	if(nmean == 0) fulloffset = coffset = offset[chan][0];
	else {
	  long sum = (version < 2) ? 0 : nmean / 2;
	  for(i = 0; i < nmean; i++) sum += offset[chan][i];
	  if(version < 2)
	    coffset = sum / nmean;
	  else {
	    fulloffset = sum / nmean;
	    coffset = ROUNDEDSHIFTDOWN(fulloffset, bitshift);
	  }
	}

	/* find the best model */
	if(bitshift == NBITPERLONG && version >= 2) {
	  fnd = FN_ZERO;
	}
	else {
	  int maxresnbitshift, snrbitshift, extrabitshift;
	  float sigpow, nn;

	  if(maxnlpc == 0)
	    fnd = wav2poly(cbuffer, blocksize,coffset,version,&sigbit,&resbit);
	  else
	    nlpc = wav2lpc(cbuffer, blocksize, coffset, qlpc, maxnlpc, version,
      	                 &sigbit, &resbit);

	  if(resbit > 0.0)
	    resn = floor(resbit + 0.5);
	  else
	    resn = 0;
	  
	  maxresnbitshift = floor(resbit - maxresn[chan] + 0.5);
	  sigpow          = exp(2.0 * M_LN2 * sigbit) / (0.5 * M_LN2 * M_LN2);
          nn              = 12.0 * sigpow / pow(10.0, minsnr / 10.0);
 	  snrbitshift     =(nn>25.0/12.0)?floor(0.5*log(nn-25.0/12.0)/M_LN2):0;
	  extrabitshift   = MAX(maxresnbitshift, snrbitshift);

	  if(extrabitshift > resn) extrabitshift = resn;

	  if(extrabitshift > 0) {
	    long offset = (1L << (extrabitshift - 1));
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = (cbuffer[i] + offset) >> extrabitshift;
	    bitshift += extrabitshift;
	    if(version >= 2)
	      coffset = ROUNDEDSHIFTDOWN(fulloffset, bitshift);
	    resn -= extrabitshift;
	  }
        }

      	/* store mean value if appropriate : N.B. Duplicated code */
      	if(nmean > 0) {
      	  long sum = (version < 2) ? 0 : blocksize / 2;

	  for(i = 0; i < blocksize; i++)
	    sum += cbuffer[i];

      	  for(i = 1; i < nmean; i++)
	    offset[chan][i - 1] = offset[chan][i];
	  if(version < 2)
	    offset[chan][nmean - 1] = sum / blocksize;
	  else
	    offset[chan][nmean - 1] = (sum / blocksize) << bitshift;
      	}

	if(bitshift != lastbitshift) {
	  uvar_put((ulong) FN_BITSHIFT, FNSIZE, fileo);
	  uvar_put((ulong) bitshift, BITSHIFTSIZE, fileo);
	  lastbitshift = bitshift;
	}
	
	if(fnd == FN_ZERO) {
	  uvar_put((ulong) fnd, FNSIZE, fileo);
	}
	else if(maxnlpc == 0) {
	  uvar_put((ulong) fnd, FNSIZE, fileo);
	  uvar_put((ulong) resn, ENERGYSIZE, fileo);

	  switch(fnd) {
	  case FN_DIFF0:
	    for(i = 0; i < blocksize; i++)
	      VAR_PUT(cbuffer[i] - coffset, resn, fileo);
	    break;
	  case FN_DIFF1:
	    for(i = 0; i < blocksize; i++)
	      VAR_PUT(cbuffer[i] - cbuffer[i - 1], resn, fileo);
	    break;
	  case FN_DIFF2:
	    for(i = 0; i < blocksize; i++)
	      VAR_PUT(cbuffer[i] - 2 * cbuffer[i - 1] + cbuffer[i - 2],
		      resn, fileo);
	    break;
	  case FN_DIFF3:
	    for(i = 0; i < blocksize; i++)
	      VAR_PUT(cbuffer[i] - 3 * (cbuffer[i - 1] - cbuffer[i - 2]) -
		      cbuffer[i - 3], resn, fileo);
	    break;
	  }
	}
	else {
	  uvar_put((ulong) FN_QLPC, FNSIZE, fileo);
	  uvar_put((ulong) resn, ENERGYSIZE, fileo);
	  uvar_put((ulong) nlpc, LPCQSIZE, fileo);
	  for(i = 0; i < nlpc; i++)
	    var_put((long) qlpc[i], LPCQUANT, fileo);

	  /* deduct mean from everything */
	  for(i = -nlpc; i < blocksize; i++)
	    cbuffer[i] -= coffset;

	  /* use the quantised LPC coefficients to generate the residual */
	  for(i = 0; i < blocksize; i++) {
	    int j;
	    long sum = lpcqoffset;
	    long *obuffer = &(cbuffer[i - 1]);

	    for(j = 0; j < nlpc; j++)
	      sum += qlpc[j] * obuffer[-j];
	    var_put(cbuffer[i] - (sum >> LPCQUANT), resn, fileo);
	  }

	  /* add mean back to those samples that will be wrapped */
	  for(i = blocksize - nwrap; i < blocksize; i++)
	    cbuffer[i] += coffset;
	}

	/* do the wrap */
	for(i = -nwrap; i < 0; i++)
	  cbuffer[i] = cbuffer[i + blocksize];
      }
    }

    /* wind up */
    fread_type_quit();
    uvar_put((ulong) FN_QUIT, FNSIZE, fileo);
    var_put_quit(fileo);

    /* and free the space used */
    free((char*) buffer);
    free((char*) buffer1);
    free((char*) offset);
    if(maxnlpc > 0)
      free((char*) qlpc);
    /* BUG FIXED BY JON FISCUS */
    if (maxresn != (float *)0)
	free((char *)maxresn);

  }
  else {
    /***********************/
    /* EXTRACT starts here */
    /***********************/

    int i, cmd;

    /* firstly skip the number of bytes requested in the command line */
    for(i = 0; i < nskip; i++) {
      int byte = getc(filei);
      if(byte == EOF)
	usage_exit(1, "File too short for requested alignment\n");
      putc_exit(byte, fileo);
    }

    /* read magic number */
#ifdef STRICT_FORMAT_COMPATABILITY
    if(FORMAT_VERSION < 2) {
      for(i = 0; i < strlen(magic); i++)
	if(getc_exit(filei) != magic[i])
	  usage_exit(1, "Bad magic number\n");

      /* get version number */
      version = getc_exit(filei);
    }
    else
#endif /* STRICT_FORMAT_COMPATABILITY */
    {
      int nscan = 0;

      version = MAX_VERSION + 1;
      while(version > MAX_VERSION) {
	int byte = getc(filei);
	if(byte == EOF)
	  usage_exit(1, "No magic number\n");
	if(magic[nscan] != '\0' && byte == magic[nscan]) nscan++;
	else if(magic[nscan] == '\0' && byte <= MAX_VERSION) version = byte;
	else {
	  for(i = 0; i < nscan; i++) putc_exit(magic[i], fileo);
	  if(byte == magic[0]) nscan = 1;
	  else {
	    putc_exit(byte, fileo);
	    nscan = 0;
	  }
	  version = MAX_VERSION + 1;
	}
      }
    }

    /* check version number */
    if(version > MAX_SUPPORTED_VERSION)
      update_exit(1, "can't decode version %d\n", version);

    /* set up the default nmean, ignoring the command line state */
    nmean = (version < 2) ? DEFAULT_V0NMEAN : DEFAULT_V2NMEAN;

    /* initialise the variable length file read for the compressed stream */
    var_get_init();

    /* initialise the fixed length file write for the uncompressed stream */
    fwrite_type_init();

    /* get file type and set up appropriately, ignoring command line state */
    ftype = UINT_GET(TYPESIZE, filei);
    if(ftype >= TYPE_EOF)
      update_exit(1, "can't decode file type %d\n", ftype);

    nchan = UINT_GET(CHANSIZE, filei);

    /* get blocksize if version > 0 */
    if(version > 0) {
      blocksize = UINT_GET((int) (log((double) DEFAULT_BLOCK_SIZE) / M_LN2),
			   filei);
      maxnlpc = UINT_GET(LPCQSIZE, filei);
      nmean = UINT_GET(0, filei);
      nskip = UINT_GET(NSKIPSIZE, filei);
      for(i = 0; i < nskip; i++) {
	int byte = uvar_get(XBYTESIZE, filei);
	putc_exit(byte, fileo);
      }
    }
    else
      blocksize = DEFAULT_BLOCK_SIZE;
    nwrap = MAX(NWRAP, maxnlpc);

    /* grab some space for the input buffer */
    buffer  = long2d((ulong) nchan, (ulong) (blocksize + nwrap));
    offset  = long2d((ulong) nchan, (ulong) MAX(1, nmean));

    for(chan = 0; chan < nchan; chan++) {
      for(i = 0; i < nwrap; i++) buffer[chan][i] = 0;
      buffer[chan] += nwrap;
    }

    if(maxnlpc > 0)
      qlpc = (int*) pmalloc((ulong) (maxnlpc * sizeof(*qlpc)));

    if(version > 1)
      lpcqoffset = V2LPCQOFFSET;

    init_offset(offset, nchan, MAX(1, nmean), ftype);

    /* get commands from file and execute them */
    chan = 0;
    while((cmd = uvar_get(FNSIZE, filei)) != FN_QUIT)
      switch(cmd) {
      case FN_ZERO:
      case FN_DIFF0:
      case FN_DIFF1:
      case FN_DIFF2:
      case FN_DIFF3:
      case FN_QLPC:
  	{ long coffset, *cbuffer = buffer[chan];
	  int resn = 0, nlpc, j;

	  if(cmd != FN_ZERO) {
	    resn = uvar_get(ENERGYSIZE, filei);
	    /* this is a hack as version 0 differed in definition of var_get */
	    if(version == 0) resn--;
	  }

  	  /* find mean offset : N.B. this code duplicated */
  	  if(nmean == 0) coffset = offset[chan][0];
	  else {
	    long sum = (version < 2) ? 0 : nmean / 2;
	    for(i = 0; i < nmean; i++) sum += offset[chan][i];
	    if(version < 2)
	      coffset = sum / nmean;
	    else
	      coffset = ROUNDEDSHIFTDOWN(sum / nmean, bitshift);
	  }

	  switch(cmd) {
	  case FN_ZERO:
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = 0;
	    break;
	  case FN_DIFF0:
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = var_get(resn, filei) + coffset;
	    break;
	  case FN_DIFF1:
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = var_get(resn, filei) + cbuffer[i - 1];
	    break;
	  case FN_DIFF2:
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = var_get(resn, filei) + (2 * cbuffer[i - 1] -
						   cbuffer[i - 2]);
	    break;
	  case FN_DIFF3:
	    for(i = 0; i < blocksize; i++)
	      cbuffer[i] = var_get(resn, filei) + 3 * (cbuffer[i - 1] -
					     cbuffer[i - 2]) + cbuffer[i - 3];
	    break;
	   case FN_QLPC:
	    nlpc = uvar_get(LPCQSIZE, filei);

	    for(i = 0; i < nlpc; i++)
	      qlpc[i] = var_get(LPCQUANT, filei);
	    for(i = 0; i < nlpc; i++)
	      cbuffer[i - nlpc] -= coffset;
	    for(i = 0; i < blocksize; i++) {
	      long sum = lpcqoffset;

	      for(j = 0; j < nlpc; j++)
		sum += qlpc[j] * cbuffer[i - j - 1];
	      cbuffer[i] = var_get(resn, filei) + (sum >> LPCQUANT);
	    }
	    if(coffset != 0)
	      for(i = 0; i < blocksize; i++)
		cbuffer[i] += coffset;
	    break;
	  }

  	  /* store mean value if appropriate : N.B. Duplicated code */
      	  if(nmean > 0) {
      	    long sum = (version < 2) ? 0 : blocksize / 2;

	    for(i = 0; i < blocksize; i++)
	      sum += cbuffer[i];

      	    for(i = 1; i < nmean; i++)
	      offset[chan][i - 1] = offset[chan][i];
	    if(version < 2)
	      offset[chan][nmean - 1] = sum / blocksize;
	    else
	      offset[chan][nmean - 1] = (sum / blocksize) << bitshift;
      	  }

	  /* do the wrap */
	  for(i = -nwrap; i < 0; i++) cbuffer[i] = cbuffer[i + blocksize];

	  fix_bitshift(cbuffer, blocksize, bitshift, ftype);

	  if(chan == nchan - 1)
	    fwrite_type(buffer, ftype, nchan, blocksize, fileo);
	  chan = (chan + 1) % nchan;
	}
	break;
      case FN_BLOCKSIZE:
	blocksize = UINT_GET((int) (log((double) blocksize) / M_LN2), filei);
	break;
      case FN_BITSHIFT:
	bitshift = uvar_get(BITSHIFTSIZE, filei);
	break;
      default:
	update_exit(1, "sanity check fails trying to decode function: %d\n",
		    cmd);
      }

    /* wind up */
    var_get_quit();
    fwrite_type_quit();

    free((char*) buffer);
    free((char*) offset);
    if(maxnlpc > 0)
      free((char*) qlpc);
  }

  /* close the files if this function opened them */
  if(filei != stdi) fclose(filei);
  if(fileo != stdo) fclose(fileo);

  /* make the compressed file look like the original if possible */
  if((filei != stdi) && (fileo != stdo))
    (void) dupfileinfo(filenamei, filenameo);

  if(tmpfilename != NULL)
    free(tmpfilename);

  if(nfilename == 1)
    if(unlink(filenamei))
      perror_exit("unlink(\"%s\")", filenamei);

  /* quit happy */
  return(0);
}

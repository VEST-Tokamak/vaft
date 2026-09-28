C     No-op stand-ins for the PGPLOT routines GENRAY calls.
C
C     Written for VAFT (install/install_genray.sh); not part of GENRAY or
C     PGPLOT. GENRAY draws its diagnostic plots through PGPLOT, which few
C     machines have and which VAFT never reads: the adapter consumes
C     genray.nc. Linking these instead drops the plots and nothing else.
C     PGOPEN reports success so GENRAY's plotting branches return normally.
C
C     A routine missing here is a link error, not a silent omission: add it
C     with the PGPLOT argument list and an empty body.
      INTEGER FUNCTION PGOPEN(DEVICE)
      CHARACTER*(*) DEVICE
      PGOPEN = 1
      END
      SUBROUTINE PGBOX(XOPT,XTICK,NXSUB,YOPT,YTICK,NYSUB)
      CHARACTER*(*) XOPT, YOPT
      END
      SUBROUTINE PGCLOS
      END
      SUBROUTINE PGCONL(A,IDIM,JDIM,I1,I2,J1,J2,C,TR,LABEL,
     &                  INTVAL,MININT)
      CHARACTER*(*) LABEL
      END
      SUBROUTINE PGCONT(A,IDIM,JDIM,I1,I2,J1,J2,C,NC,TR)
      END
      SUBROUTINE PGEND
      END
      SUBROUTINE PGENV(XMIN,XMAX,YMIN,YMAX,JUST,AXIS)
      END
      SUBROUTINE PGLAB(XLBL,YLBL,TOPLBL)
      CHARACTER*(*) XLBL, YLBL, TOPLBL
      END
      SUBROUTINE PGLINE(N,XPTS,YPTS)
      END
      SUBROUTINE PGMTXT(SIDE,DISP,COORD,FJUST,TEXT)
      CHARACTER*(*) SIDE, TEXT
      END
      SUBROUTINE PGPAGE
      END
      SUBROUTINE PGPT(N,XPTS,YPTS,SYMBOL)
      END
      SUBROUTINE PGSCH(SIZE)
      END
      SUBROUTINE PGSCI(CI)
      END
      SUBROUTINE PGSLS(LS)
      END
      SUBROUTINE PGSLW(LW)
      END
      SUBROUTINE PGSVP(XLEFT,XRIGHT,YBOT,YTOP)
      END
      SUBROUTINE PGSWIN(X1,X2,Y1,Y2)
      END
      SUBROUTINE PGTEXT(X,Y,TEXT)
      CHARACTER*(*) TEXT
      END

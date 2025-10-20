%----------------------------------------------------------------
% Simple document class for bachelor/master theses.
% 2019 by Sebastian Otte
%----------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2019/03/13 bachelor/master thesis class]

\newif\ifthesistype@bsc
\newif\ifthesistype@msc


\thesistype@bsctrue

\DeclareOption{bsc}{
    \thesistype@bsctrue
    \thesistype@mscfalse
}
\DeclareOption{msc}{
    \thesistype@bscfalse
    \thesistype@msctrue
}


\PassOptionsToClass{%
    a4paper,
    12pt,
    headsepline,
    numbers=noenddot,
    BCOR12mm,
    chapterprefix,
    appendixprefix,
    index=totoc,
    bibliography=totoc,
    parskip=never
}{scrbook}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}


%\makeatletter

\ProcessOptions*\relax

\ifthesistype@bsc
\newcommand{\thethesistype}{Bachelorarbeit}
\fi

\ifthesistype@msc
\newcommand{\thethesistype}{Masterarbeit}
\fi

\LoadClass{scrbook}

\RequirePackage{amsmath}
\RequirePackage{amssymb} 
\RequirePackage{nicefrac}
\RequirePackage{upgreek}

\RequirePackage{rotating}
\usepackage{etoolbox}

\RequirePackage[labelfont=bf, format=plain,font=small]{caption}

\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{natbib}
\RequirePackage{supertabular}

\RequirePackage[utf8]{inputenc}


\RequirePackage[english, ngerman]{babel}

\RequirePackage{hyperref}
\RequirePackage{url}

\RequirePackage{makeidx}

%\RequirePackage{layouts}
%\setlength{\stockheight}{11in}

\RequirePackage{tocloft}

\pagestyle{headings}

\setkomafont{descriptionlabel}{\bfseries}
\setkomafont{sectioning}{\bfseries}

\addtokomafont{chapterprefix}{\Large}
\renewcommand*{\chapterformat}{%
    \chapapp~\thechapter\autodot\enskip\par%
}

\newcommand{\authorfirstname}[1]{\def\theauthorfirstname{#1}}
\newcommand{\authorlastname}[1]{\def\theauthorlastname{#1}}
\newcommand{\course}[1]{\def\thecourse{#1}}
\newcommand{\editingperiod}[1]{\def\theeditingperiod{#1}}

\newcommand{\reviewer}[2]{%
\def\thereviewercaption{#1}
\def\thereviewer{#2}
}

\newcommand{\supervisor}[2]{%
\def\thesupervisorcaption{#1}
\def\thesupervisor{#2}
}
%\newcommand{\supervisor}[q]{\def\thesupervisor{#1}}

\newcommand\referentname[1]{%
{\large #1}
}

\newcommand\referentinstitute[1]{%
{\footnotesize #1}   
}

\author{\theauthorfirstname \theauthorlastname}


\renewcommand{\maketitle}{%
\begin{titlepage}
    \begin{center}
        \LARGE Eberhard Karls Universität Tübingen\\
        \large Mathematisch-Naturwissenschaftliche Fakultät\\
        Wilhelm-Schickard-Institut für Informatik\\
        [3cm]
        \LARGE {\thethesistype} {\thecourse}\\
        [1.0cm]
        \Large\textbf{\@title}\\
        [1.5cm]
        \large {\theauthorfirstname} {\theauthorlastname}\\
        [0.5cm]
        \normalsize\@date\\
        \vfill
        \ifdefined\thereviewer
        \small\textbf{\thereviewercaption}\\[0.3cm]
        \thereviewer\\[1cm]
        \fi
        \ifdefined\thesupervisor
        \small\textbf{\thesupervisorcaption}\\[0.3cm]
        \thesupervisor
        \else
        \fi
    \end{center}
\newpage

\thispagestyle{empty}
\vspace*{\fill}
\noindent\textbf{{\theauthorlastname}, {\theauthorfirstname}:}\\
\emph{\@title}\\
{\thethesistype} {\thecourse}\\
Eberhard Karls Universität Tübingen\\
Bearbeitungszeitraum: \theeditingperiod
\end{titlepage}
}
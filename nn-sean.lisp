(defparameter *debug* nil)

(defun dprint (some-variable &optional (additional-message '()))
	(if *debug*
		(progn 
			(if additional-message (print additional-message) nil) 
			(print some-variable))
		some-variable))

(defun shuffle (lis)
  "Shuffles a list.  Non-destructive.  O(length lis), so
pretty efficient.  Returns the shuffled version of the list."
  (let ((vec (apply #'vector lis)) bag (len (length lis)))
    (dotimes (x len)
      (let ((i (random (- len x))))
	(rotatef (svref vec i) (svref vec (- len x 1)))
	(push (svref vec (- len x 1)) bag)))
    bag))   ;; 65 s-expressions, by the way


(defparameter *verify* t)

;;; hmmm, openmcl keeps signalling an error of a different kind
;;; when I throw an error -- a bug in openmcl?  dunno...
(defun throw-error (str)
  (error (make-condition 'simple-error :format-control str)))

(defun verify-equal (funcname &rest matrices)
  ;; we presume they're rectangular -- else we're REALLY in trouble!
  (when *verify*
    (unless (and
	     (apply #'= (mapcar #'length matrices))
	     (apply #'= (mapcar #'length (mapcar #'first matrices))))
      (throw-error (format t "In ~s, matrix dimensions not equal: ~s"
			   funcname
			   (mapcar #'(lambda (mat) (list (length mat) 'by (length (first mat))))
				   matrices))))))

(defun verify-multiplicable (matrix1 matrix2)
  ;; we presume they're rectangular -- else we're REALLY in trouble!
  (when *verify*
    (if (/= (length (first matrix1)) (length matrix2))
	(throw-error (format t "In multiply, matrix dimensions not valid: ~s"
			     (list (list (length matrix1) 'by (length (first matrix1)))
				   (list (length matrix2) 'by (length (first matrix2)))))))))


;; Basic Operations

(defun map-m (function &rest matrices)
  "Maps function over elements in matrices, returning a new matrix"
  (apply #'verify-equal 'map-m  matrices)
  (apply #'mapcar #'(lambda (&rest vectors)       ;; for each matrix...
		      (apply #'mapcar #'(lambda (&rest elts)     ;; for each vector...
					  (apply function elts))
			     vectors)) 
	 matrices))   ;; pretty :-)

(defun transpose (matrix)
  "Transposes a matrix"
  (apply #'mapcar #'list matrix))  ;; cool, no?

(defun make-matrix (i j func)
  "Builds a matrix with i rows and j columns,
    with each element initialized by calling (func)"
  (map-m func (make-list i :initial-element (make-list j :initial-element nil))))

(defun make-random-matrix (i j val)
  "Builds a matrix with i rows and j columns,
    with each element initialized to a random
    floating-point number between -val and val"
  (make-matrix i j #'(lambda (x)
		       (declare (ignore x))  ;; quiets warnings about x not being used
		       (- (random (* 2.0 val)) val))))

(defun e (matrix i j)
  "Returns the element at row i and column j in matrix"
  ;; 1-based, not zero-based.  This is because it's traditional
  ;; for the top-left element in a matrix to be element (1,1),
  ;; NOT (0,0).  Sorry about that.  :-)
  (elt (elt matrix (1- i)) (1- j)))

(defun print-matrix (matrix)
  "Prints a matrix in a pleasing form, then returns matrix"
  (mapcar #'(lambda (vector) (format t "~%~{~8,4,,F~}" vector)) matrix) matrix)

;;; Matrix Multiplication

(defun multiply2 (matrix1 matrix2)
  "Multiplies matrix1 by matrix2 
    -- don't use this, use multiply instead"
  (verify-multiplicable matrix1 matrix2)
  (let ((tmatrix2 (transpose matrix2)))
    (mapcar #'(lambda (vector1)
		(mapcar #'(lambda (vector2)
			    (apply #'+ (mapcar #'* vector1 vector2))) tmatrix2))
	    matrix1)))  ;; pretty :-)

(defun multiply (matrix1 matrix2 &rest matrices)
  "Multiplies matrices together"
  (reduce #'multiply2 (cons matrix1 (cons matrix2 matrices))))

;;; Element-by-element operations

(defun add (matrix1 matrix2 &rest matrices)
  "Adds matrices together, returning a new matrix"
  (apply #'verify-equal 'add matrix1 matrix2 matrices)
  (apply #'map-m #'+ matrix1 matrix2 matrices))

(defun e-multiply (matrix1 matrix2 &rest matrices)
  "Multiplies corresponding elements in matrices together, 
        returning a new matrix"
  (apply #'verify-equal 'e-multiply matrix1 matrix2 matrices)
  (apply #'map-m #'* matrix1 matrix2 matrices))

(defun subtract (matrix1 matrix2 &rest matrices)
  "Subtracts matrices from the first matrix, returning a new matrix."
  (let ((all (cons matrix1 (cons matrix2 matrices))))
    (apply #'verify-equal 'subtract all)
    (apply #'map-m #'- all)))

(defun scalar-add (scalar matrix)
  "Adds scalar to each element in matrix, returning a new matrix"
  (map-m #'(lambda (elt) (+ scalar elt)) matrix))

(defun scalar-multiply (scalar matrix)
  "Multiplies each element in matrix by scalar, returning a new matrix"
  (map-m #'(lambda (elt) (* scalar elt)) matrix))

;;; This function could
;;; be done trivially with (scalar-add scalar (scalar-multiply -1 matrix))
(defun subtract-from-scalar (scalar matrix)
  "Subtracts each element in the matrix from scalar, returning a new matrix"
  (map-m #'(lambda (elt) (- scalar elt)) matrix))






;;; Functions you need to implement

;; IMPLEMENT THIS FUNCTION

(defun sigmoid (value)
	(/ 1 (+ 1 (exp (* -1 value)))));; 1/(1+e^value)

;; output and correct-output are both column-vectors

;; IMPLEMENT THIS FUNCTION
;;  "Returns (as a scalar value) the error between the output and correct vectors"
(defun net-error (output correct-output)
		(mapcar '- output correct-output)
  )


;; a single datum is of the form
;; (--input-column-vector--  -- output-column-vector--)
;;
;; Notice that this is different from the raw datum provided in the problems below.
;; You can convert the raw datum to this column-vector form using CONVERT-datum

;; IMPLEMENT THIS FUNCTION
;; "Returns as a vector the output of the OUTPUT units when presented
;;the datum as input."
(defun forward-propagate (input layers)
	(dprint "forward propagate:")
	(dprint input)
	;; this is recursive purely for the sake of being 'lispy'
	(if layers
		;;do the multiplication of the first layer, keep popin recursively until....
		(list input (forward-propagate   (map-m #'sigmoid (multiply (pop layers) input)) layers))
		;;...there are no more layers left. just return input given to us, it was the multiply
		input))

;; IMPLEMENT THIS FUNCTION
;;"Back-propagates a datum through the V and W matrices,
;;returning a list consisting of new, modified V and W matrices."
  ;; Consider using let*
  ;; let* is like let, except that it lets you initialize local
  ;; variables in the context of earlier local variables in the
  ;; same let* statement.
(defun back-propagate (layer-outputs layers desired-output alpha)
  (dprint "BACK-prop desired-output:")
  (dprint desired-output)
  layers
  )



;; "If option is t, then prints x, else doesn't print it.
;;In any case, returns x"
  ;;; perhaps this might be a useful function for you
(defun optionally-print (x option)
  
  (if option (print x) x))


(defparameter *a-good-minimum-error* 1.0e-9)



;; datum is of the form:
;; (
;;  (--input-column-vector--  --output-column-vector--)
;;  (--input-column-vector--  --output-column-vector--)
;;  ...
;; )
;;
;;
;; Notice that this is different from the raw datum provided in the problems below.
;; You can convert the raw datum to this column-vector form using CONVERT-datum


;;; DAVID's helpers for net-build ;;;
(defun init-neural-layers (num-neurons input-size num-layers output-size initial-bounds)
	
	(let (layers '())
		;; this line is really long and annoying, i have to do it 3 times, there has to be a way to avoid this? 
		;;besides being long, this lines initializes the first layer of nn, leading form input to hidden layers
		(dprint "info stuff:")
		(dprint num-neurons)
		(dprint input-size)
		(dprint initial-bounds)
		(dprint "layer dimensions:")
		(setf layers (append layers (list (make-random-matrix   (dprint num-neurons)  (dprint input-size) initial-bounds))))
		(dprint ":")
		;;does the same thing as above, but between each hidden layer (!!not used for base assignment)
		(dotimes (i (- num-layers 1)) 
				(setf layers (append layers (list (make-random-matrix  (dprint num-neurons)  (dprint num-neurons) initial-bounds)))))
		(dprint ":")
		;;creates the matrix hidden layer to output
		(setf layers (append layers (list  (make-random-matrix (dprint output-size) (dprint num-neurons) initial-bounds))))))

  ;;returns a list of two elements, representing input and output sizes. Example: nand returns (2 1)
(defun extract-input-and-output-sizes (datum)
 ;; this is parsed based on how the datum is formatted in sean's datumsets (AFTER convert-datum)
 (dprint "input, outputs is returning")
 (dprint (second (first datum)))
 (dprint (list (length (first  (first datum)))  (length  (second (first datum))))))
;;; IMPLEMENT THIS FUNCTION

;;"Builds a neural network with num-hidden-units and the appropriate number
;;of input and output units based on the datum.  Each element should be a random
;;value between -(INITIAL-BOUNDS) and +(INITIAL-BOUNDS).
;;
;;Then performs the following loop MAX-ITERATIONS times, or until the error condition
;;is met (see below):
;;
;;   1. For each datum element in a randomized version of the datum, perform
;;      backpropagation.
;;   2. Every modulo iterations,
;;          For every datum element in the datum, perform forward propagation and
;;          A.  If print-all-errors is true, then print the error for each element
;;          B.  At any rate, always print the worst error and the mean error
;;          C.  If the worst error is better (lower) than A-GOOD-MINIMUM-ERROR,
;;              quit all loops and prepare to exit the function --
;;              the error condition was met.
;;The function should return a list of two items: the final V matrix
;;and the final W matrix of the learned network."
(defun net-build (datum num-hidden-units alpha initial-bounds max-iterations modulo &optional print-all-errors)
  	;; use my two helper functions, extract-input-and-output-sizes to get appropriate sizes
  	;; use init-neural-layers to actually make the matrix
  	(dprint "initial bounds is:")
  	(dprint initial-bounds)
  	;;(print (extract-input-and-output-sizes datum))
	(let ((i-o-size (extract-input-and-output-sizes datum)))
		(dprint i-o-size "i-o-size is:")
		
		(init-neural-layers num-hidden-units (first i-o-size) 1 (second i-o-size) initial-bounds)))





;; For this function, you should pass in the datum just like it's defined
;; in the example problems below (that is, not in the "column vector" format
;; used by NET-BUILD.  Of course, if you need to call NET_BUILD from this function
;; you can alway convert this datum to column-vector format using CONVERT-datum within
;; the SIMPLE-GENERALIZATION function.
;;
;; Yes, this is ridiculously inconsistent.  Deal with it.  :-)

;;; IMPLEMENT THIS FUNCTION
;; "Given a set of datum, trains a neural network on the first half
;;of the datum, then tests generalization on the second half, returning
;;the average error among the samples in the second half.  Don't print any errors,
;;and use a modulo of MAX-ITERATIONS."
(defun simple-generalization (datum num-hidden-units alpha initial-bounds max-iterations)
 	
 	
	;;need to get num inputs, num outputs from datum.
	;;let layer-datum 
  )
(defun full-data-training (datum num-hidden-units alpha initial-bounds max-iterations)
	
	;;(print (forward-propagate (first (first (convert-datum *xor*))) (net-build (convert-datum *xor*) 3 .2 9 90 2)))
	;;net-build (datum num-hidden-units alpha initial-bounds max-iterations modulo &optional print-all-errors)

	(let ((layers (net-build datum num-hidden-units alpha initial-bounds max-iterations 1)))
		(loop for i from 1 to max-iterations do(progn
			 (shuffle datum)
			 
			(loop for a from 1 to (- (length datum) 1) do(progn
				(let ((layer-outputs (forward-propagate (first (nth a (dprint datum "hey this is the dataset i'm grabbing the nth of:"))) layers )))
					(setf layers (back-propagate 
						(dprint layer-outputs "supplied layer outputs to back-prop:") layers (second (nth a datum)) alpha))))
)))))


;; For this function, you should pass in the datum just like it's defined
;; in the example problems below (that is, not in the "column vector" format
;; used by NET-BUILD.  Of course, if you need to call NET_BUILD from this function
;; you can alway convert this datum to column-vector format using CONVERT-datum within
;; the SIMPLE-GENERALIZATION function.
;;
;; Yes, this is ridiculously inconsistent.  Deal with it.  :-)


;;; IMPLEMENT THIS FUNCTION FOR EXTRA CREDIT
;;"Given a set of datum, performs k-fold validation on this datum for
;;the provided value of k, by training the network on (k-1)/k of the datum,
;;then testing generalization on the remaining 1/k of the datum.  This is
;;done k times for different 1/k chunks (and building k different networks).
;;The average error among all tested samples is returned.  Don't print any errors,
;;and use a modulo of MAX-ITERATIONS."
(defun k-fold-validation (datum k))




;;;; Some useful preprocessing functions

(defun scale-list (lis)
  "Scales a list so the minimum value is 0.1 and the maximum value is 0.9.  Don't use this function, it's just used by scale-datum."
  (let ((min (reduce #'min lis))
	(max (reduce #'max lis)))
    (mapcar (lambda (elt) (+ 0.1 (* 0.8 (/ (- elt min) (- max min)))))
	    lis)))

(defun scale-datum (lis)
  "Scales all the attributes in a list of samples of the form ((attributes) (outputs))"
  (transpose (list (transpose (mapcar #'scale-list (transpose (mapcar #'first lis))))
		   (transpose (mapcar #'scale-list (transpose (mapcar #'second lis)))))))

(defun convert-datum (raw-datum)
  "Converts raw datum into column-vector datum of the form that
can be fed into NET-LEARN.  Also adds a bias unit of 0.5 to the input."
  (mapcar #'(lambda (datum)
	      (mapcar #'(lambda (vec)
			  (mapcar #'list vec))
		      (list (cons 0.5 (first datum))
			    (second datum))))
	  raw-datum))

(defun average (lis)
  "Computes the average over a list of numbers.  Returns 0 if the list length is 0."
  (if (= (length lis) 0)
      0
      (/ (reduce #'+ lis) (length lis))))


;;; here are the inputs and outputs of your three problems to test
;;; the net function on.


(defparameter *nand*
  '(((0.1 0.1) (0.9))
    ((0.9 0.1) (0.9))
    ((0.1 0.9) (0.9))
    ((0.9 0.9) (0.1))))


(defparameter *xor*
  '(((0.1 0.1) (0.1))
    ((0.9 0.1) (0.9))
    ((0.1 0.9) (0.9))
    ((0.9 0.9) (0.1))))




;; I converted the attribute values as follows:
;; Democrat -> 0.9, Republican -> 0.1
;; y -> 0.9, n -> 0.1, no-vote -> 0.5

;; The output is democrat or republican

;; the input is the following votes:
					;   1. handicapped-infants: 2 (y,n)
					;   2. water-project-cost-sharing: 2 (y,n)
					;   3. adoption-of-the-budget-resolution: 2 (y,n)
					;   4. physician-fee-freeze: 2 (y,n)
					;   5. el-salvador-aid: 2 (y,n)
					;   6. religious-groups-in-schools: 2 (y,n)
					;   7. anti-satellite-test-ban: 2 (y,n)
					;   8. aid-to-nicaraguan-contras: 2 (y,n)
					;   9. mx-missile: 2 (y,n)
					;  10. immigration: 2 (y,n)
					;  11. synfuels-corporation-cutback: 2 (y,n)
					;  12. education-spending: 2 (y,n)
					;  13. superfund-right-to-sue: 2 (y,n)
					;  14. crime: 2 (y,n)
					;  15. duty-free-exports: 2 (y,n)
					;  16. export-administration-act-south-africa: 2 (y,n)


(defparameter *voting-records*
  '(((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.5 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.5 0.9 0.9 0.5 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.5 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.5 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.5 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.5 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.5 0.5 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5 0.9 0.9 0.5 0.5) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.5 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.5 0.9 0.9 0.5 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5 0.5 0.1 0.5) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.1 0.5) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.9 0.5 0.9 0.9 0.9 0.5 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.5 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.5 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.5 0.5 0.9 0.9) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.5 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1))
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.1 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.5 0.5) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.5 0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.5 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.5 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.9 0.9 0.9 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.5 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.5 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.5 0.5 0.1 0.9 0.5 0.5 0.5 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.5 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.5 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.9 0.1 0.9 0.9 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.5 0.1 0.1 0.1 0.1 0.9 0.5 0.1) (0.9)) 
    ((0.5 0.5 0.5 0.5 0.1 0.9 0.9 0.9 0.9 0.9 0.5 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.9 0.5 0.5) (0.1)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.5 0.9 0.1 0.5 0.5 0.9 0.9 0.9 0.9 0.5 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.5 0.5 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.9) (0.1)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.5 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.5 0.9 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.5 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.5 0.5 0.5 0.5 0.5 0.5) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.5 0.5 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.5 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.5 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.5 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.5 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.5 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.5 0.5 0.5 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.5 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.5 0.9 0.9 0.5 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.5 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.5 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.5 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.5 0.5 0.1 0.1 0.1 0.5 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.9 0.5 0.5 0.5 0.5 0.5 0.5 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.5 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.5 0.9 0.5 0.5) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.5 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.5) (0.9)) 
    ((0.1 0.5 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.5 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.5 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.5 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.5 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.9 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.1 0.1 0.1 0.1 0.5 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.5 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.5 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.5 0.9 0.9 0.9 0.1 0.5 0.5 0.1 0.5 0.5 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.9) (0.1)) 
    ((0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.5 0.1 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1) (0.1)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.5) (0.1)) 
    ((0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.5 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.5 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.5 0.9 0.5 0.9 0.9 0.9 0.1 0.9 0.9 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.5 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.5 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.5 0.9 0.5 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.5 0.1 0.9 0.1 0.9 0.5 0.5 0.5 0.5) (0.1)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.5 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.5 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.1) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.5 0.1 0.1 0.1 0.1 0.5 0.5 0.9 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.1) (0.9)) 
    ((0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.5 0.9 0.1 0.5 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.5 0.9 0.1 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.5 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.5 0.1 0.9 0.9 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.9 0.5 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.5 0.9 0.5 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.5 0.5 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.5 0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.5 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.5 0.9 0.9 0.9 0.1 0.9 0.5 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.5 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.5 0.9 0.1 0.1) (0.9)) 
    ((0.1 0.9 0.9 0.5 0.9 0.9 0.1 0.9 0.1 0.9 0.5 0.1 0.9 0.9 0.5 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1) (0.9)) 
    ((0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.5 0.1 0.1 0.5 0.5 0.9 0.5 0.5 0.5) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.9 0.5 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1) (0.9)) 
    ((0.1 0.5 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.1 0.1 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.5) (0.9))
    ((0.5 0.5 0.1 0.1 0.5 0.9 0.5 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.5) (0.9)) 
    ((0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.1) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.5 0.5 0.5 0.5 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.5) (0.1)) 
    ((0.9 0.9 0.5 0.5 0.5 0.9 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.5 0.1 0.1 0.1 0.9 0.1 0.1 0.9 0.5 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.1 0.9 0.5 0.1 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.1 0.9 0.5 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.5 0.1 0.1 0.5 0.5 0.5 0.9 0.1 0.5) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.5 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.5 0.9 0.1 0.1) (0.1)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.5 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.5 0.5 0.5 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.5 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.1 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.9 0.9 0.1 0.1 0.5 0.9 0.9 0.9 0.9 0.9 0.1 0.5 0.9 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.1 0.1 0.1 0.9 0.5) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.5 0.5 0.5 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.1 0.9 0.9) (0.9)) 
    ((0.9 0.1 0.9 0.1 0.5 0.1 0.9 0.9 0.9 0.9 0.1 0.9 0.1 0.5 0.9 0.9) (0.9)) 
    ((0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.9 0.9 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.9 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.9) (0.9)) 
    ((0.1 0.5 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.9 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.1 0.1 0.9 0.9 0.9 0.5 0.5 0.5 0.5 0.1 0.9 0.9 0.9 0.1 0.9) (0.1)) 
    ((0.1 0.9 0.1 0.9 0.9 0.9 0.1 0.1 0.1 0.9 0.1 0.9 0.9 0.9 0.5 0.1) (0.1))))



;;;; This datum isn't normalized.  I suggest scaling the datum to between 0.1 and 0.9
;;;; before trying to learn it (scale each attribute independent of other attributes)
;;;;
;;;; The attributes are:
;;;;
;;;;    1. cylinders:     multi-valued discrete
;;;;    2. displacement:  continuous
;;;;    3. horsepower:    continuous
;;;;    4. weight:        continuous
;;;;    5. acceleration:  continuous
;;;;    6. model year:    multi-valued discrete
;;;;    7. origin:        multi-valued discrete   (1 = USA, 2 = Europe, 3 = Asia)
;;;;
;;;; The output is:
;;;;    1. mpg:           continuous
;;;;


(defparameter *mpg*
  '(((8 307.0 130.0 3504 12.0 70 1) (18.0))    ;;  chevrolet chevelle malibu
    ((8 350.0 165.0 3693 11.5 70 1) (15.0))    ;;  buick skylark 320
    ((8 318.0 150.0 3436 11.0 70 1) (18.0))    ;;  plymouth satellite
    ((8 304.0 150.0 3433 12.0 70 1) (16.0))    ;;  amc rebel sst
    ((8 302.0 140.0 3449 10.5 70 1) (17.0))    ;;  ford torino
    ((8 429.0 198.0 4341 10.0 70 1) (15.0))    ;;  ford galaxie 500
    ((8 454.0 220.0 4354 9.0 70 1) (14.0))    ;;  chevrolet impala
    ((8 440.0 215.0 4312 8.5 70 1) (14.0))    ;;  plymouth fury iii
    ((8 455.0 225.0 4425 10.0 70 1) (14.0))    ;;  pontiac catalina
    ((8 390.0 190.0 3850 8.5 70 1) (15.0))    ;;  amc ambassador dpl
    ((8 383.0 170.0 3563 10.0 70 1) (15.0))    ;;  dodge challenger se
    ((8 340.0 160.0 3609 8.0 70 1) (14.0))    ;;  plymouth 'cuda 340
    ((8 400.0 150.0 3761 9.5 70 1) (15.0))    ;;  chevrolet monte carlo
    ((8 455.0 225.0 3086 10.0 70 1) (14.0))    ;;  buick estate wagon (sw)
    ((4 113.0 95.0 2372 15.0 70 3) (24.0))    ;;  toyota corona mark ii
    ((6 198.0 95.0 2833 15.5 70 1) (22.0))    ;;  plymouth duster
    ((6 199.0 97.0 2774 15.5 70 1) (18.0))    ;;  amc hornet
    ((6 200.0 85.0 2587 16.0 70 1) (21.0))    ;;  ford maverick
    ((4 97.0 88.0 2130 14.5 70 3) (27.0))    ;;  datsun pl510
    ((4 97.0 46.0 1835 20.5 70 2) (26.0))    ;;  volkswagen 1131 deluxe sedan
    ((4 110.0 87.0 2672 17.5 70 2) (25.0))    ;;  peugeot 504
    ((4 107.0 90.0 2430 14.5 70 2) (24.0))    ;;  audi 100 ls
    ((4 104.0 95.0 2375 17.5 70 2) (25.0))    ;;  saab 99e
    ((4 121.0 113.0 2234 12.5 70 2) (26.0))    ;;  bmw 2002
    ((6 199.0 90.0 2648 15.0 70 1) (21.0))    ;;  amc gremlin
    ((8 360.0 215.0 4615 14.0 70 1) (10.0))    ;;  ford f250
    ((8 307.0 200.0 4376 15.0 70 1) (10.0))    ;;  chevy c20
    ((8 318.0 210.0 4382 13.5 70 1) (11.0))    ;;  dodge d200
    ((8 304.0 193.0 4732 18.5 70 1) (9.0))    ;;  hi 1200d
    ((4 97.0 88.0 2130 14.5 71 3) (27.0))    ;;  datsun pl510
    ((4 140.0 90.0 2264 15.5 71 1) (28.0))    ;;  chevrolet vega 2300
    ((4 113.0 95.0 2228 14.0 71 3) (25.0))    ;;  toyota corona
    ((6 232.0 100.0 2634 13.0 71 1) (19.0))    ;;  amc gremlin
    ((6 225.0 105.0 3439 15.5 71 1) (16.0))    ;;  plymouth satellite custom
    ((6 250.0 100.0 3329 15.5 71 1) (17.0))    ;;  chevrolet chevelle malibu
    ((6 250.0 88.0 3302 15.5 71 1) (19.0))    ;;  ford torino 500
    ((6 232.0 100.0 3288 15.5 71 1) (18.0))    ;;  amc matador
    ((8 350.0 165.0 4209 12.0 71 1) (14.0))    ;;  chevrolet impala
    ((8 400.0 175.0 4464 11.5 71 1) (14.0))    ;;  pontiac catalina brougham
    ((8 351.0 153.0 4154 13.5 71 1) (14.0))    ;;  ford galaxie 500
    ((8 318.0 150.0 4096 13.0 71 1) (14.0))    ;;  plymouth fury iii
    ((8 383.0 180.0 4955 11.5 71 1) (12.0))    ;;  dodge monaco (sw)
    ((8 400.0 170.0 4746 12.0 71 1) (13.0))    ;;  ford country squire (sw)
    ((8 400.0 175.0 5140 12.0 71 1) (13.0))    ;;  pontiac safari (sw)
    ((6 258.0 110.0 2962 13.5 71 1) (18.0))    ;;  amc hornet sportabout (sw)
    ((4 140.0 72.0 2408 19.0 71 1) (22.0))    ;;  chevrolet vega (sw)
    ((6 250.0 100.0 3282 15.0 71 1) (19.0))    ;;  pontiac firebird
    ((6 250.0 88.0 3139 14.5 71 1) (18.0))    ;;  ford mustang
    ((4 122.0 86.0 2220 14.0 71 1) (23.0))    ;;  mercury capri 2000
    ((4 116.0 90.0 2123 14.0 71 2) (28.0))    ;;  opel 1900
    ((4 79.0 70.0 2074 19.5 71 2) (30.0))    ;;  peugeot 304
    ((4 88.0 76.0 2065 14.5 71 2) (30.0))    ;;  fiat 124b
    ((4 71.0 65.0 1773 19.0 71 3) (31.0))    ;;  toyota corolla 1200
    ((4 72.0 69.0 1613 18.0 71 3) (35.0))    ;;  datsun 1200
    ((4 97.0 60.0 1834 19.0 71 2) (27.0))    ;;  volkswagen model 111
    ((4 91.0 70.0 1955 20.5 71 1) (26.0))    ;;  plymouth cricket
    ((4 113.0 95.0 2278 15.5 72 3) (24.0))    ;;  toyota corona hardtop
    ((4 97.5 80.0 2126 17.0 72 1) (25.0))    ;;  dodge colt hardtop
    ((4 97.0 54.0 2254 23.5 72 2) (23.0))    ;;  volkswagen type 3
    ((4 140.0 90.0 2408 19.5 72 1) (20.0))    ;;  chevrolet vega
    ((4 122.0 86.0 2226 16.5 72 1) (21.0))    ;;  ford pinto runabout
    ((8 350.0 165.0 4274 12.0 72 1) (13.0))    ;;  chevrolet impala
    ((8 400.0 175.0 4385 12.0 72 1) (14.0))    ;;  pontiac catalina
    ((8 318.0 150.0 4135 13.5 72 1) (15.0))    ;;  plymouth fury iii
    ((8 351.0 153.0 4129 13.0 72 1) (14.0))    ;;  ford galaxie 500
    ((8 304.0 150.0 3672 11.5 72 1) (17.0))    ;;  amc ambassador sst
    ((8 429.0 208.0 4633 11.0 72 1) (11.0))    ;;  mercury marquis
    ((8 350.0 155.0 4502 13.5 72 1) (13.0))    ;;  buick lesabre custom
    ((8 350.0 160.0 4456 13.5 72 1) (12.0))    ;;  oldsmobile delta 88 royale
    ((8 400.0 190.0 4422 12.5 72 1) (13.0))    ;;  chrysler newport royal
    ((3 70.0 97.0 2330 13.5 72 3) (19.0))    ;;  mazda rx2 coupe
    ((8 304.0 150.0 3892 12.5 72 1) (15.0))    ;;  amc matador (sw)
    ((8 307.0 130.0 4098 14.0 72 1) (13.0))    ;;  chevrolet chevelle concours (sw)
    ((8 302.0 140.0 4294 16.0 72 1) (13.0))    ;;  ford gran torino (sw)
    ((8 318.0 150.0 4077 14.0 72 1) (14.0))    ;;  plymouth satellite custom (sw)
    ((4 121.0 112.0 2933 14.5 72 2) (18.0))    ;;  volvo 145e (sw)
    ((4 121.0 76.0 2511 18.0 72 2) (22.0))    ;;  volkswagen 411 (sw)
    ((4 120.0 87.0 2979 19.5 72 2) (21.0))    ;;  peugeot 504 (sw)
    ((4 96.0 69.0 2189 18.0 72 2) (26.0))    ;;  renault 12 (sw)
    ((4 122.0 86.0 2395 16.0 72 1) (22.0))    ;;  ford pinto (sw)
    ((4 97.0 92.0 2288 17.0 72 3) (28.0))    ;;  datsun 510 (sw)
    ((4 120.0 97.0 2506 14.5 72 3) (23.0))    ;;  toyouta corona mark ii (sw)
    ((4 98.0 80.0 2164 15.0 72 1) (28.0))    ;;  dodge colt (sw)
    ((4 97.0 88.0 2100 16.5 72 3) (27.0))    ;;  toyota corolla 1600 (sw)
    ((8 350.0 175.0 4100 13.0 73 1) (13.0))    ;;  buick century 350
    ((8 304.0 150.0 3672 11.5 73 1) (14.0))    ;;  amc matador
    ((8 350.0 145.0 3988 13.0 73 1) (13.0))    ;;  chevrolet malibu
    ((8 302.0 137.0 4042 14.5 73 1) (14.0))    ;;  ford gran torino
    ((8 318.0 150.0 3777 12.5 73 1) (15.0))    ;;  dodge coronet custom
    ((8 429.0 198.0 4952 11.5 73 1) (12.0))    ;;  mercury marquis brougham
    ((8 400.0 150.0 4464 12.0 73 1) (13.0))    ;;  chevrolet caprice classic
    ((8 351.0 158.0 4363 13.0 73 1) (13.0))    ;;  ford ltd
    ((8 318.0 150.0 4237 14.5 73 1) (14.0))    ;;  plymouth fury gran sedan
    ((8 440.0 215.0 4735 11.0 73 1) (13.0))    ;;  chrysler new yorker brougham
    ((8 455.0 225.0 4951 11.0 73 1) (12.0))    ;;  buick electra 225 custom
    ((8 360.0 175.0 3821 11.0 73 1) (13.0))    ;;  amc ambassador brougham
    ((6 225.0 105.0 3121 16.5 73 1) (18.0))    ;;  plymouth valiant
    ((6 250.0 100.0 3278 18.0 73 1) (16.0))    ;;  chevrolet nova custom
    ((6 232.0 100.0 2945 16.0 73 1) (18.0))    ;;  amc hornet
    ((6 250.0 88.0 3021 16.5 73 1) (18.0))    ;;  ford maverick
    ((6 198.0 95.0 2904 16.0 73 1) (23.0))    ;;  plymouth duster
    ((4 97.0 46.0 1950 21.0 73 2) (26.0))    ;;  volkswagen super beetle
    ((8 400.0 150.0 4997 14.0 73 1) (11.0))    ;;  chevrolet impala
    ((8 400.0 167.0 4906 12.5 73 1) (12.0))    ;;  ford country
    ((8 360.0 170.0 4654 13.0 73 1) (13.0))    ;;  plymouth custom suburb
    ((8 350.0 180.0 4499 12.5 73 1) (12.0))    ;;  oldsmobile vista cruiser
    ((6 232.0 100.0 2789 15.0 73 1) (18.0))    ;;  amc gremlin
    ((4 97.0 88.0 2279 19.0 73 3) (20.0))    ;;  toyota carina
    ((4 140.0 72.0 2401 19.5 73 1) (21.0))    ;;  chevrolet vega
    ((4 108.0 94.0 2379 16.5 73 3) (22.0))    ;;  datsun 610
    ((3 70.0 90.0 2124 13.5 73 3) (18.0))    ;;  maxda rx3
    ((4 122.0 85.0 2310 18.5 73 1) (19.0))    ;;  ford pinto
    ((6 155.0 107.0 2472 14.0 73 1) (21.0))    ;;  mercury capri v6
    ((4 98.0 90.0 2265 15.5 73 2) (26.0))    ;;  fiat 124 sport coupe
    ((8 350.0 145.0 4082 13.0 73 1) (15.0))    ;;  chevrolet monte carlo s
    ((8 400.0 230.0 4278 9.5 73 1) (16.0))    ;;  pontiac grand prix
    ((4 68.0 49.0 1867 19.5 73 2) (29.0))    ;;  fiat 128
    ((4 116.0 75.0 2158 15.5 73 2) (24.0))    ;;  opel manta
    ((4 114.0 91.0 2582 14.0 73 2) (20.0))    ;;  audi 100ls
    ((4 121.0 112.0 2868 15.5 73 2) (19.0))    ;;  volvo 144ea
    ((8 318.0 150.0 3399 11.0 73 1) (15.0))    ;;  dodge dart custom
    ((4 121.0 110.0 2660 14.0 73 2) (24.0))    ;;  saab 99le
    ((6 156.0 122.0 2807 13.5 73 3) (20.0))    ;;  toyota mark ii
    ((8 350.0 180.0 3664 11.0 73 1) (11.0))    ;;  oldsmobile omega
    ((6 198.0 95.0 3102 16.5 74 1) (20.0))    ;;  plymouth duster
    ((6 232.0 100.0 2901 16.0 74 1) (19.0))    ;;  amc hornet
    ((6 250.0 100.0 3336 17.0 74 1) (15.0))    ;;  chevrolet nova
    ((4 79.0 67.0 1950 19.0 74 3) (31.0))    ;;  datsun b210
    ((4 122.0 80.0 2451 16.5 74 1) (26.0))    ;;  ford pinto
    ((4 71.0 65.0 1836 21.0 74 3) (32.0))    ;;  toyota corolla 1200
    ((4 140.0 75.0 2542 17.0 74 1) (25.0))    ;;  chevrolet vega
    ((6 250.0 100.0 3781 17.0 74 1) (16.0))    ;;  chevrolet chevelle malibu classic
    ((6 258.0 110.0 3632 18.0 74 1) (16.0))    ;;  amc matador
    ((6 225.0 105.0 3613 16.5 74 1) (18.0))    ;;  plymouth satellite sebring
    ((8 302.0 140.0 4141 14.0 74 1) (16.0))    ;;  ford gran torino
    ((8 350.0 150.0 4699 14.5 74 1) (13.0))    ;;  buick century luxus (sw)
    ((8 318.0 150.0 4457 13.5 74 1) (14.0))    ;;  dodge coronet custom (sw)
    ((8 302.0 140.0 4638 16.0 74 1) (14.0))    ;;  ford gran torino (sw)
    ((8 304.0 150.0 4257 15.5 74 1) (14.0))    ;;  amc matador (sw)
    ((4 98.0 83.0 2219 16.5 74 2) (29.0))    ;;  audi fox
    ((4 79.0 67.0 1963 15.5 74 2) (26.0))    ;;  volkswagen dasher
    ((4 97.0 78.0 2300 14.5 74 2) (26.0))    ;;  opel manta
    ((4 76.0 52.0 1649 16.5 74 3) (31.0))    ;;  toyota corona
    ((4 83.0 61.0 2003 19.0 74 3) (32.0))    ;;  datsun 710
    ((4 90.0 75.0 2125 14.5 74 1) (28.0))    ;;  dodge colt
    ((4 90.0 75.0 2108 15.5 74 2) (24.0))    ;;  fiat 128
    ((4 116.0 75.0 2246 14.0 74 2) (26.0))    ;;  fiat 124 tc
    ((4 120.0 97.0 2489 15.0 74 3) (24.0))    ;;  honda civic
    ((4 108.0 93.0 2391 15.5 74 3) (26.0))    ;;  subaru
    ((4 79.0 67.0 2000 16.0 74 2) (31.0))    ;;  fiat x1.9
    ((6 225.0 95.0 3264 16.0 75 1) (19.0))    ;;  plymouth valiant custom
    ((6 250.0 105.0 3459 16.0 75 1) (18.0))    ;;  chevrolet nova
    ((6 250.0 72.0 3432 21.0 75 1) (15.0))    ;;  mercury monarch
    ((6 250.0 72.0 3158 19.5 75 1) (15.0))    ;;  ford maverick
    ((8 400.0 170.0 4668 11.5 75 1) (16.0))    ;;  pontiac catalina
    ((8 350.0 145.0 4440 14.0 75 1) (15.0))    ;;  chevrolet bel air
    ((8 318.0 150.0 4498 14.5 75 1) (16.0))    ;;  plymouth grand fury
    ((8 351.0 148.0 4657 13.5 75 1) (14.0))    ;;  ford ltd
    ((6 231.0 110.0 3907 21.0 75 1) (17.0))    ;;  buick century
    ((6 250.0 105.0 3897 18.5 75 1) (16.0))    ;;  chevroelt chevelle malibu
    ((6 258.0 110.0 3730 19.0 75 1) (15.0))    ;;  amc matador
    ((6 225.0 95.0 3785 19.0 75 1) (18.0))    ;;  plymouth fury
    ((6 231.0 110.0 3039 15.0 75 1) (21.0))    ;;  buick skyhawk
    ((8 262.0 110.0 3221 13.5 75 1) (20.0))    ;;  chevrolet monza 2+2
    ((8 302.0 129.0 3169 12.0 75 1) (13.0))    ;;  ford mustang ii
    ((4 97.0 75.0 2171 16.0 75 3) (29.0))    ;;  toyota corolla
    ((4 140.0 83.0 2639 17.0 75 1) (23.0))    ;;  ford pinto
    ((6 232.0 100.0 2914 16.0 75 1) (20.0))    ;;  amc gremlin
    ((4 140.0 78.0 2592 18.5 75 1) (23.0))    ;;  pontiac astro
    ((4 134.0 96.0 2702 13.5 75 3) (24.0))    ;;  toyota corona
    ((4 90.0 71.0 2223 16.5 75 2) (25.0))    ;;  volkswagen dasher
    ((4 119.0 97.0 2545 17.0 75 3) (24.0))    ;;  datsun 710
    ((6 171.0 97.0 2984 14.5 75 1) (18.0))    ;;  ford pinto
    ((4 90.0 70.0 1937 14.0 75 2) (29.0))    ;;  volkswagen rabbit
    ((6 232.0 90.0 3211 17.0 75 1) (19.0))    ;;  amc pacer
    ((4 115.0 95.0 2694 15.0 75 2) (23.0))    ;;  audi 100ls
    ((4 120.0 88.0 2957 17.0 75 2) (23.0))    ;;  peugeot 504
    ((4 121.0 98.0 2945 14.5 75 2) (22.0))    ;;  volvo 244dl
    ((4 121.0 115.0 2671 13.5 75 2) (25.0))    ;;  saab 99le
    ((4 91.0 53.0 1795 17.5 75 3) (33.0))    ;;  honda civic cvcc
    ((4 107.0 86.0 2464 15.5 76 2) (28.0))    ;;  fiat 131
    ((4 116.0 81.0 2220 16.9 76 2) (25.0))    ;;  opel 1900
    ((4 140.0 92.0 2572 14.9 76 1) (25.0))    ;;  capri ii
    ((4 98.0 79.0 2255 17.7 76 1) (26.0))    ;;  dodge colt
    ((4 101.0 83.0 2202 15.3 76 2) (27.0))    ;;  renault 12tl
    ((8 305.0 140.0 4215 13.0 76 1) (17.5))    ;;  chevrolet chevelle malibu classic
    ((8 318.0 150.0 4190 13.0 76 1) (16.0))    ;;  dodge coronet brougham
    ((8 304.0 120.0 3962 13.9 76 1) (15.5))    ;;  amc matador
    ((8 351.0 152.0 4215 12.8 76 1) (14.5))    ;;  ford gran torino
    ((6 225.0 100.0 3233 15.4 76 1) (22.0))    ;;  plymouth valiant
    ((6 250.0 105.0 3353 14.5 76 1) (22.0))    ;;  chevrolet nova
    ((6 200.0 81.0 3012 17.6 76 1) (24.0))    ;;  ford maverick
    ((6 232.0 90.0 3085 17.6 76 1) (22.5))    ;;  amc hornet
    ((4 85.0 52.0 2035 22.2 76 1) (29.0))    ;;  chevrolet chevette
    ((4 98.0 60.0 2164 22.1 76 1) (24.5))    ;;  chevrolet woody
    ((4 90.0 70.0 1937 14.2 76 2) (29.0))    ;;  vw rabbit
    ((4 91.0 53.0 1795 17.4 76 3) (33.0))    ;;  honda civic
    ((6 225.0 100.0 3651 17.7 76 1) (20.0))    ;;  dodge aspen se
    ((6 250.0 78.0 3574 21.0 76 1) (18.0))    ;;  ford granada ghia
    ((6 250.0 110.0 3645 16.2 76 1) (18.5))    ;;  pontiac ventura sj
    ((6 258.0 95.0 3193 17.8 76 1) (17.5))    ;;  amc pacer d/l
    ((4 97.0 71.0 1825 12.2 76 2) (29.5))    ;;  volkswagen rabbit
    ((4 85.0 70.0 1990 17.0 76 3) (32.0))    ;;  datsun b-210
    ((4 97.0 75.0 2155 16.4 76 3) (28.0))    ;;  toyota corolla
    ((4 140.0 72.0 2565 13.6 76 1) (26.5))    ;;  ford pinto
    ((4 130.0 102.0 3150 15.7 76 2) (20.0))    ;;  volvo 245
    ((8 318.0 150.0 3940 13.2 76 1) (13.0))    ;;  plymouth volare premier v8
    ((4 120.0 88.0 3270 21.9 76 2) (19.0))    ;;  peugeot 504
    ((6 156.0 108.0 2930 15.5 76 3) (19.0))    ;;  toyota mark ii
    ((6 168.0 120.0 3820 16.7 76 2) (16.5))    ;;  mercedes-benz 280s
    ((8 350.0 180.0 4380 12.1 76 1) (16.5))    ;;  cadillac seville
    ((8 350.0 145.0 4055 12.0 76 1) (13.0))    ;;  chevy c10
    ((8 302.0 130.0 3870 15.0 76 1) (13.0))    ;;  ford f108
    ((8 318.0 150.0 3755 14.0 76 1) (13.0))    ;;  dodge d100
    ((4 98.0 68.0 2045 18.5 77 3) (31.5))    ;;  honda accord cvcc
    ((4 111.0 80.0 2155 14.8 77 1) (30.0))    ;;  buick opel isuzu deluxe
    ((4 79.0 58.0 1825 18.6 77 2) (36.0))    ;;  renault 5 gtl
    ((4 122.0 96.0 2300 15.5 77 1) (25.5))    ;;  plymouth arrow gs
    ((4 85.0 70.0 1945 16.8 77 3) (33.5))    ;;  datsun f-10 hatchback
    ((8 305.0 145.0 3880 12.5 77 1) (17.5))    ;;  chevrolet caprice classic
    ((8 260.0 110.0 4060 19.0 77 1) (17.0))    ;;  oldsmobile cutlass supreme
    ((8 318.0 145.0 4140 13.7 77 1) (15.5))    ;;  dodge monaco brougham
    ((8 302.0 130.0 4295 14.9 77 1) (15.0))    ;;  mercury cougar brougham
    ((6 250.0 110.0 3520 16.4 77 1) (17.5))    ;;  chevrolet concours
    ((6 231.0 105.0 3425 16.9 77 1) (20.5))    ;;  buick skylark
    ((6 225.0 100.0 3630 17.7 77 1) (19.0))    ;;  plymouth volare custom
    ((6 250.0 98.0 3525 19.0 77 1) (18.5))    ;;  ford granada
    ((8 400.0 180.0 4220 11.1 77 1) (16.0))    ;;  pontiac grand prix lj
    ((8 350.0 170.0 4165 11.4 77 1) (15.5))    ;;  chevrolet monte carlo landau
    ((8 400.0 190.0 4325 12.2 77 1) (15.5))    ;;  chrysler cordoba
    ((8 351.0 149.0 4335 14.5 77 1) (16.0))    ;;  ford thunderbird
    ((4 97.0 78.0 1940 14.5 77 2) (29.0))    ;;  volkswagen rabbit custom
    ((4 151.0 88.0 2740 16.0 77 1) (24.5))    ;;  pontiac sunbird coupe
    ((4 97.0 75.0 2265 18.2 77 3) (26.0))    ;;  toyota corolla liftback
    ((4 140.0 89.0 2755 15.8 77 1) (25.5))    ;;  ford mustang ii 2+2
    ((4 98.0 63.0 2051 17.0 77 1) (30.5))    ;;  chevrolet chevette
    ((4 98.0 83.0 2075 15.9 77 1) (33.5))    ;;  dodge colt m/m
    ((4 97.0 67.0 1985 16.4 77 3) (30.0))    ;;  subaru dl
    ((4 97.0 78.0 2190 14.1 77 2) (30.5))    ;;  volkswagen dasher
    ((6 146.0 97.0 2815 14.5 77 3) (22.0))    ;;  datsun 810
    ((4 121.0 110.0 2600 12.8 77 2) (21.5))    ;;  bmw 320i
    ((3 80.0 110.0 2720 13.5 77 3) (21.5))    ;;  mazda rx-4
    ((4 90.0 48.0 1985 21.5 78 2) (43.1))    ;;  volkswagen rabbit custom diesel
    ((4 98.0 66.0 1800 14.4 78 1) (36.1))    ;;  ford fiesta
    ((4 78.0 52.0 1985 19.4 78 3) (32.8))    ;;  mazda glc deluxe
    ((4 85.0 70.0 2070 18.6 78 3) (39.4))    ;;  datsun b210 gx
    ((4 91.0 60.0 1800 16.4 78 3) (36.1))    ;;  honda civic cvcc
    ((8 260.0 110.0 3365 15.5 78 1) (19.9))    ;;  oldsmobile cutlass salon brougham
    ((8 318.0 140.0 3735 13.2 78 1) (19.4))    ;;  dodge diplomat
    ((8 302.0 139.0 3570 12.8 78 1) (20.2))    ;;  mercury monarch ghia
    ((6 231.0 105.0 3535 19.2 78 1) (19.2))    ;;  pontiac phoenix lj
    ((6 200.0 95.0 3155 18.2 78 1) (20.5))    ;;  chevrolet malibu
    ((6 200.0 85.0 2965 15.8 78 1) (20.2))    ;;  ford fairmont (auto)
    ((4 140.0 88.0 2720 15.4 78 1) (25.1))    ;;  ford fairmont (man)
    ((6 225.0 100.0 3430 17.2 78 1) (20.5))    ;;  plymouth volare
    ((6 232.0 90.0 3210 17.2 78 1) (19.4))    ;;  amc concord
    ((6 231.0 105.0 3380 15.8 78 1) (20.6))    ;;  buick century special
    ((6 200.0 85.0 3070 16.7 78 1) (20.8))    ;;  mercury zephyr
    ((6 225.0 110.0 3620 18.7 78 1) (18.6))    ;;  dodge aspen
    ((6 258.0 120.0 3410 15.1 78 1) (18.1))    ;;  amc concord d/l
    ((8 305.0 145.0 3425 13.2 78 1) (19.2))    ;;  chevrolet monte carlo landau
    ((6 231.0 165.0 3445 13.4 78 1) (17.7))    ;;  buick regal sport coupe (turbo)
    ((8 302.0 139.0 3205 11.2 78 1) (18.1))    ;;  ford futura
    ((8 318.0 140.0 4080 13.7 78 1) (17.5))    ;;  dodge magnum xe
    ((4 98.0 68.0 2155 16.5 78 1) (30.0))    ;;  chevrolet chevette
    ((4 134.0 95.0 2560 14.2 78 3) (27.5))    ;;  toyota corona
    ((4 119.0 97.0 2300 14.7 78 3) (27.2))    ;;  datsun 510
    ((4 105.0 75.0 2230 14.5 78 1) (30.9))    ;;  dodge omni
    ((4 134.0 95.0 2515 14.8 78 3) (21.1))    ;;  toyota celica gt liftback
    ((4 156.0 105.0 2745 16.7 78 1) (23.2))    ;;  plymouth sapporo
    ((4 151.0 85.0 2855 17.6 78 1) (23.8))    ;;  oldsmobile starfire sx
    ((4 119.0 97.0 2405 14.9 78 3) (23.9))    ;;  datsun 200-sx
    ((5 131.0 103.0 2830 15.9 78 2) (20.3))    ;;  audi 5000
    ((6 163.0 125.0 3140 13.6 78 2) (17.0))    ;;  volvo 264gl
    ((4 121.0 115.0 2795 15.7 78 2) (21.6))    ;;  saab 99gle
    ((6 163.0 133.0 3410 15.8 78 2) (16.2))    ;;  peugeot 604sl
    ((4 89.0 71.0 1990 14.9 78 2) (31.5))    ;;  volkswagen scirocco
    ((4 98.0 68.0 2135 16.6 78 3) (29.5))    ;;  honda accord lx
    ((6 231.0 115.0 3245 15.4 79 1) (21.5))    ;;  pontiac lemans v6
    ((6 200.0 85.0 2990 18.2 79 1) (19.8))    ;;  mercury zephyr 6
    ((4 140.0 88.0 2890 17.3 79 1) (22.3))    ;;  ford fairmont 4
    ((6 232.0 90.0 3265 18.2 79 1) (20.2))    ;;  amc concord dl 6
    ((6 225.0 110.0 3360 16.6 79 1) (20.6))    ;;  dodge aspen 6
    ((8 305.0 130.0 3840 15.4 79 1) (17.0))    ;;  chevrolet caprice classic
    ((8 302.0 129.0 3725 13.4 79 1) (17.6))    ;;  ford ltd landau
    ((8 351.0 138.0 3955 13.2 79 1) (16.5))    ;;  mercury grand marquis
    ((8 318.0 135.0 3830 15.2 79 1) (18.2))    ;;  dodge st. regis
    ((8 350.0 155.0 4360 14.9 79 1) (16.9))    ;;  buick estate wagon (sw)
    ((8 351.0 142.0 4054 14.3 79 1) (15.5))    ;;  ford country squire (sw)
    ((8 267.0 125.0 3605 15.0 79 1) (19.2))    ;;  chevrolet malibu classic (sw)
    ((8 360.0 150.0 3940 13.0 79 1) (18.5))    ;;  chrysler lebaron town @ country (sw)
    ((4 89.0 71.0 1925 14.0 79 2) (31.9))    ;;  vw rabbit custom
    ((4 86.0 65.0 1975 15.2 79 3) (34.1))    ;;  maxda glc deluxe
    ((4 98.0 80.0 1915 14.4 79 1) (35.7))    ;;  dodge colt hatchback custom
    ((4 121.0 80.0 2670 15.0 79 1) (27.4))    ;;  amc spirit dl
    ((5 183.0 77.0 3530 20.1 79 2) (25.4))    ;;  mercedes benz 300d
    ((8 350.0 125.0 3900 17.4 79 1) (23.0))    ;;  cadillac eldorado
    ((4 141.0 71.0 3190 24.8 79 2) (27.2))    ;;  peugeot 504
    ((8 260.0 90.0 3420 22.2 79 1) (23.9))    ;;  oldsmobile cutlass salon brougham
    ((4 105.0 70.0 2200 13.2 79 1) (34.2))    ;;  plymouth horizon
    ((4 105.0 70.0 2150 14.9 79 1) (34.5))    ;;  plymouth horizon tc3
    ((4 85.0 65.0 2020 19.2 79 3) (31.8))    ;;  datsun 210
    ((4 91.0 69.0 2130 14.7 79 2) (37.3))    ;;  fiat strada custom
    ((4 151.0 90.0 2670 16.0 79 1) (28.4))    ;;  buick skylark limited
    ((6 173.0 115.0 2595 11.3 79 1) (28.8))    ;;  chevrolet citation
    ((6 173.0 115.0 2700 12.9 79 1) (26.8))    ;;  oldsmobile omega brougham
    ((4 151.0 90.0 2556 13.2 79 1) (33.5))    ;;  pontiac phoenix
    ((4 98.0 76.0 2144 14.7 80 2) (41.5))    ;;  vw rabbit
    ((4 89.0 60.0 1968 18.8 80 3) (38.1))    ;;  toyota corolla tercel
    ((4 98.0 70.0 2120 15.5 80 1) (32.1))    ;;  chevrolet chevette
    ((4 86.0 65.0 2019 16.4 80 3) (37.2))    ;;  datsun 310
    ((4 151.0 90.0 2678 16.5 80 1) (28.0))    ;;  chevrolet citation
    ((4 140.0 88.0 2870 18.1 80 1) (26.4))    ;;  ford fairmont
    ((4 151.0 90.0 3003 20.1 80 1) (24.3))    ;;  amc concord
    ((6 225.0 90.0 3381 18.7 80 1) (19.1))    ;;  dodge aspen
    ((4 97.0 78.0 2188 15.8 80 2) (34.3))    ;;  audi 4000
    ((4 134.0 90.0 2711 15.5 80 3) (29.8))    ;;  toyota corona liftback
    ((4 120.0 75.0 2542 17.5 80 3) (31.3))    ;;  mazda 626
    ((4 119.0 92.0 2434 15.0 80 3) (37.0))    ;;  datsun 510 hatchback
    ((4 108.0 75.0 2265 15.2 80 3) (32.2))    ;;  toyota corolla
    ((4 86.0 65.0 2110 17.9 80 3) (46.6))    ;;  mazda glc
    ((4 156.0 105.0 2800 14.4 80 1) (27.9))    ;;  dodge colt
    ((4 85.0 65.0 2110 19.2 80 3) (40.8))    ;;  datsun 210
    ((4 90.0 48.0 2085 21.7 80 2) (44.3))    ;;  vw rabbit c (diesel)
    ((4 90.0 48.0 2335 23.7 80 2) (43.4))    ;;  vw dasher (diesel)
    ((5 121.0 67.0 2950 19.9 80 2) (36.4))    ;;  audi 5000s (diesel)
    ((4 146.0 67.0 3250 21.8 80 2) (30.0))    ;;  mercedes-benz 240d
    ((4 91.0 67.0 1850 13.8 80 3) (44.6))    ;;  honda civic 1500 gl
    ((4 97.0 67.0 2145 18.0 80 3) (33.8))    ;;  subaru dl
    ((4 89.0 62.0 1845 15.3 80 2) (29.8))    ;;  vokswagen rabbit
    ((6 168.0 132.0 2910 11.4 80 3) (32.7))    ;;  datsun 280-zx
    ((3 70.0 100.0 2420 12.5 80 3) (23.7))    ;;  mazda rx-7 gs
    ((4 122.0 88.0 2500 15.1 80 2) (35.0))    ;;  triumph tr7 coupe
    ((4 107.0 72.0 2290 17.0 80 3) (32.4))    ;;  honda accord
    ((4 135.0 84.0 2490 15.7 81 1) (27.2))    ;;  plymouth reliant
    ((4 151.0 84.0 2635 16.4 81 1) (26.6))    ;;  buick skylark
    ((4 156.0 92.0 2620 14.4 81 1) (25.8))    ;;  dodge aries wagon (sw)
    ((6 173.0 110.0 2725 12.6 81 1) (23.5))    ;;  chevrolet citation
    ((4 135.0 84.0 2385 12.9 81 1) (30.0))    ;;  plymouth reliant
    ((4 79.0 58.0 1755 16.9 81 3) (39.1))    ;;  toyota starlet
    ((4 86.0 64.0 1875 16.4 81 1) (39.0))    ;;  plymouth champ
    ((4 81.0 60.0 1760 16.1 81 3) (35.1))    ;;  honda civic 1300
    ((4 97.0 67.0 2065 17.8 81 3) (32.3))    ;;  subaru
    ((4 85.0 65.0 1975 19.4 81 3) (37.0))    ;;  datsun 210 mpg
    ((4 89.0 62.0 2050 17.3 81 3) (37.7))    ;;  toyota tercel
    ((4 91.0 68.0 1985 16.0 81 3) (34.1))    ;;  mazda glc 4
    ((4 105.0 63.0 2215 14.9 81 1) (34.7))    ;;  plymouth horizon 4
    ((4 98.0 65.0 2045 16.2 81 1) (34.4))    ;;  ford escort 4w
    ((4 98.0 65.0 2380 20.7 81 1) (29.9))    ;;  ford escort 2h
    ((4 105.0 74.0 2190 14.2 81 2) (33.0))    ;;  volkswagen jetta
    ((4 107.0 75.0 2210 14.4 81 3) (33.7))    ;;  honda prelude
    ((4 108.0 75.0 2350 16.8 81 3) (32.4))    ;;  toyota corolla
    ((4 119.0 100.0 2615 14.8 81 3) (32.9))    ;;  datsun 200sx
    ((4 120.0 74.0 2635 18.3 81 3) (31.6))    ;;  mazda 626
    ((4 141.0 80.0 3230 20.4 81 2) (28.1))    ;;  peugeot 505s turbo diesel
    ((6 145.0 76.0 3160 19.6 81 2) (30.7))    ;;  volvo diesel
    ((6 168.0 116.0 2900 12.6 81 3) (25.4))    ;;  toyota cressida
    ((6 146.0 120.0 2930 13.8 81 3) (24.2))    ;;  datsun 810 maxima
    ((6 231.0 110.0 3415 15.8 81 1) (22.4))    ;;  buick century
    ((8 350.0 105.0 3725 19.0 81 1) (26.6))    ;;  oldsmobile cutlass ls
    ((6 200.0 88.0 3060 17.1 81 1) (20.2))    ;;  ford granada gl
    ((6 225.0 85.0 3465 16.6 81 1) (17.6))    ;;  chrysler lebaron salon
    ((4 112.0 88.0 2605 19.6 82 1) (28.0))    ;;  chevrolet cavalier
    ((4 112.0 88.0 2640 18.6 82 1) (27.0))    ;;  chevrolet cavalier wagon
    ((4 112.0 88.0 2395 18.0 82 1) (34.0))    ;;  chevrolet cavalier 2-door
    ((4 112.0 85.0 2575 16.2 82 1) (31.0))    ;;  pontiac j2000 se hatchback
    ((4 135.0 84.0 2525 16.0 82 1) (29.0))    ;;  dodge aries se
    ((4 151.0 90.0 2735 18.0 82 1) (27.0))    ;;  pontiac phoenix
    ((4 140.0 92.0 2865 16.4 82 1) (24.0))    ;;  ford fairmont futura
    ((4 105.0 74.0 1980 15.3 82 2) (36.0))    ;;  volkswagen rabbit l
    ((4 91.0 68.0 2025 18.2 82 3) (37.0))    ;;  mazda glc custom l
    ((4 91.0 68.0 1970 17.6 82 3) (31.0))    ;;  mazda glc custom
    ((4 105.0 63.0 2125 14.7 82 1) (38.0))    ;;  plymouth horizon miser
    ((4 98.0 70.0 2125 17.3 82 1) (36.0))    ;;  mercury lynx l
    ((4 120.0 88.0 2160 14.5 82 3) (36.0))    ;;  nissan stanza xe
    ((4 107.0 75.0 2205 14.5 82 3) (36.0))    ;;  honda accord
    ((4 108.0 70.0 2245 16.9 82 3) (34.0))    ;;  toyota corolla
    ((4 91.0 67.0 1965 15.0 82 3) (38.0))    ;;  honda civic
    ((4 91.0 67.0 1965 15.7 82 3) (32.0))    ;;  honda civic (auto)
    ((4 91.0 67.0 1995 16.2 82 3) (38.0))    ;;  datsun 310 gx
    ((6 181.0 110.0 2945 16.4 82 1) (25.0))    ;;  buick century limited
    ((6 262.0 85.0 3015 17.0 82 1) (38.0))    ;;  oldsmobile cutlass ciera (diesel)
    ((4 156.0 92.0 2585 14.5 82 1) (26.0))    ;;  chrysler lebaron medallion
    ((6 232.0 112.0 2835 14.7 82 1) (22.0))    ;;  ford granada l
    ((4 144.0 96.0 2665 13.9 82 3) (32.0))    ;;  toyota celica gt
    ((4 135.0 84.0 2370 13.0 82 1) (36.0))    ;;  dodge charger 2.2
    ((4 151.0 90.0 2950 17.3 82 1) (27.0))    ;;  chevrolet camaro
    ((4 140.0 86.0 2790 15.6 82 1) (27.0))    ;;  ford mustang gl
    ((4 97.0 52.0 2130 24.6 82 2) (44.0))    ;;  vw pickup
    ((4 135.0 84.0 2295 11.6 82 1) (32.0))    ;;  dodge rampage
    ((4 120.0 79.0 2625 18.6 82 1) (28.0))    ;;  ford ranger
    ((4 119.0 82.0 2720 19.4 82 1) (31.0))    ;;  chevy s-10
    ))



;;;; This datum isn't normalized.  I suggest scaling the datum to between 0.1 and 0.9
;;;; before trying to learn it (scale each attribute independent of other attributes)
;;;;
;;;; The attributes are:
;;;;
;;;;    1)  Alcohol 
;;;;    2)  Malic acid 
;;;;    3)  Ash 
;;;;    4)  Alcalinity of ash 
;;;;    5)  Magnesium 
;;;;    6)  Total phenols 
;;;;    7)  Flavanoids 
;;;;    8)  Nonflavanoid phenols 
;;;;    9)  Proanthocyanins 
;;;;    10) Color intensity 
;;;;    11) Hue 
;;;;    12) OD280/OD315 of diluted wines 
;;;;    13) Proline 
;;;;
;;;; The output is:
;;;;    1. wine type    (three classes)
;;;;
;;;; In this example, I converted the three classes to binary values along three
;;;; dimensions (another common tactic for converting classification into regression
;;;; problems).  The values are:  (0.9 0.1 0.1)     (0.1 0.9 0.1)    (0.1 0.1 0.9)


(defparameter *wine* 
  '(((14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065) (0.9 0.1 0.1))
    ((13.2 1.78 2.14 11.2 100 2.65 2.76 0.26 1.28 4.38 1.05 3.4 1050) (0.9 0.1 0.1))
    ((13.16 2.36 2.67 18.6 101 2.8 3.24 0.3 2.81 5.68 1.03 3.17 1185) (0.9 0.1 0.1))
    ((14.37 1.95 2.5 16.8 113 3.85 3.49 0.24 2.18 7.8 0.86 3.45 1480) (0.9 0.1 0.1))
    ((13.24 2.59 2.87 21 118 2.8 2.69 0.39 1.82 4.32 1.04 2.93 735) (0.9 0.1 0.1))
    ((14.2 1.76 2.45 15.2 112 3.27 3.39 0.34 1.97 6.75 1.05 2.85 1450) (0.9 0.1 0.1))
    ((14.39 1.87 2.45 14.6 96 2.5 2.52 0.3 1.98 5.25 1.02 3.58 1290) (0.9 0.1 0.1))
    ((14.06 2.15 2.61 17.6 121 2.6 2.51 0.31 1.25 5.05 1.06 3.58 1295) (0.9 0.1 0.1))
    ((14.83 1.64 2.17 14 97 2.8 2.98 0.29 1.98 5.2 1.08 2.85 1045) (0.9 0.1 0.1))
    ((13.86 1.35 2.27 16 98 2.98 3.15 0.22 1.85 7.22 1.01 3.55 1045) (0.9 0.1 0.1))
    ((14.1 2.16 2.3 18 105 2.95 3.32 0.22 2.38 5.75 1.25 3.17 1510) (0.9 0.1 0.1))
    ((14.12 1.48 2.32 16.8 95 2.2 2.43 0.26 1.57 5 1.17 2.82 1280) (0.9 0.1 0.1))
    ((13.75 1.73 2.41 16 89 2.6 2.76 0.29 1.81 5.6 1.15 2.9 1320) (0.9 0.1 0.1))
    ((14.75 1.73 2.39 11.4 91 3.1 3.69 0.43 2.81 5.4 1.25 2.73 1150) (0.9 0.1 0.1))
    ((14.38 1.87 2.38 12 102 3.3 3.64 0.29 2.96 7.5 1.2 3 1547) (0.9 0.1 0.1))
    ((13.63 1.81 2.7 17.2 112 2.85 2.91 0.3 1.46 7.3 1.28 2.88 1310) (0.9 0.1 0.1))
    ((14.3 1.92 2.72 20 120 2.8 3.14 0.33 1.97 6.2 1.07 2.65 1280) (0.9 0.1 0.1))
    ((13.83 1.57 2.62 20 115 2.95 3.4 0.4 1.72 6.6 1.13 2.57 1130) (0.9 0.1 0.1))
    ((14.19 1.59 2.48 16.5 108 3.3 3.93 0.32 1.86 8.7 1.23 2.82 1680) (0.9 0.1 0.1))
    ((13.64 3.1 2.56 15.2 116 2.7 3.03 0.17 1.66 5.1 0.96 3.36 845) (0.9 0.1 0.1))
    ((14.06 1.63 2.28 16 126 3 3.17 0.24 2.1 5.65 1.09 3.71 780) (0.9 0.1 0.1))
    ((12.93 3.8 2.65 18.6 102 2.41 2.41 0.25 1.98 4.5 1.03 3.52 770) (0.9 0.1 0.1))
    ((13.71 1.86 2.36 16.6 101 2.61 2.88 0.27 1.69 3.8 1.11 4 1035) (0.9 0.1 0.1))
    ((12.85 1.6 2.52 17.8 95 2.48 2.37 0.26 1.46 3.93 1.09 3.63 1015) (0.9 0.1 0.1))
    ((13.5 1.81 2.61 20 96 2.53 2.61 0.28 1.66 3.52 1.12 3.82 845) (0.9 0.1 0.1))
    ((13.05 2.05 3.22 25 124 2.63 2.68 0.47 1.92 3.58 1.13 3.2 830) (0.9 0.1 0.1))
    ((13.39 1.77 2.62 16.1 93 2.85 2.94 0.34 1.45 4.8 0.92 3.22 1195) (0.9 0.1 0.1))
    ((13.3 1.72 2.14 17 94 2.4 2.19 0.27 1.35 3.95 1.02 2.77 1285) (0.9 0.1 0.1))
    ((13.87 1.9 2.8 19.4 107 2.95 2.97 0.37 1.76 4.5 1.25 3.4 915) (0.9 0.1 0.1))
    ((14.02 1.68 2.21 16 96 2.65 2.33 0.26 1.98 4.7 1.04 3.59 1035) (0.9 0.1 0.1))
    ((13.73 1.5 2.7 22.5 101 3 3.25 0.29 2.38 5.7 1.19 2.71 1285) (0.9 0.1 0.1))
    ((13.58 1.66 2.36 19.1 106 2.86 3.19 0.22 1.95 6.9 1.09 2.88 1515) (0.9 0.1 0.1))
    ((13.68 1.83 2.36 17.2 104 2.42 2.69 0.42 1.97 3.84 1.23 2.87 990) (0.9 0.1 0.1))
    ((13.76 1.53 2.7 19.5 132 2.95 2.74 0.5 1.35 5.4 1.25 3 1235) (0.9 0.1 0.1))
    ((13.51 1.8 2.65 19 110 2.35 2.53 0.29 1.54 4.2 1.1 2.87 1095) (0.9 0.1 0.1))
    ((13.48 1.81 2.41 20.5 100 2.7 2.98 0.26 1.86 5.1 1.04 3.47 920) (0.9 0.1 0.1))
    ((13.28 1.64 2.84 15.5 110 2.6 2.68 0.34 1.36 4.6 1.09 2.78 880) (0.9 0.1 0.1))
    ((13.05 1.65 2.55 18 98 2.45 2.43 0.29 1.44 4.25 1.12 2.51 1105) (0.9 0.1 0.1))
    ((13.07 1.5 2.1 15.5 98 2.4 2.64 0.28 1.37 3.7 1.18 2.69 1020) (0.9 0.1 0.1))
    ((14.22 3.99 2.51 13.2 128 3 3.04 0.2 2.08 5.1 0.89 3.53 760) (0.9 0.1 0.1))
    ((13.56 1.71 2.31 16.2 117 3.15 3.29 0.34 2.34 6.13 0.95 3.38 795) (0.9 0.1 0.1))
    ((13.41 3.84 2.12 18.8 90 2.45 2.68 0.27 1.48 4.28 0.91 3 1035) (0.9 0.1 0.1))
    ((13.88 1.89 2.59 15 101 3.25 3.56 0.17 1.7 5.43 0.88 3.56 1095) (0.9 0.1 0.1))
    ((13.24 3.98 2.29 17.5 103 2.64 2.63 0.32 1.66 4.36 0.82 3 680) (0.9 0.1 0.1))
    ((13.05 1.77 2.1 17 107 3 3 0.28 2.03 5.04 0.88 3.35 885) (0.9 0.1 0.1))
    ((14.21 4.04 2.44 18.9 111 2.85 2.65 0.3 1.25 5.24 0.87 3.33 1080) (0.9 0.1 0.1))
    ((14.38 3.59 2.28 16 102 3.25 3.17 0.27 2.19 4.9 1.04 3.44 1065) (0.9 0.1 0.1))
    ((13.9 1.68 2.12 16 101 3.1 3.39 0.21 2.14 6.1 0.91 3.33 985) (0.9 0.1 0.1))
    ((14.1 2.02 2.4 18.8 103 2.75 2.92 0.32 2.38 6.2 1.07 2.75 1060) (0.9 0.1 0.1))
    ((13.94 1.73 2.27 17.4 108 2.88 3.54 0.32 2.08 8.9 1.12 3.1 1260) (0.9 0.1 0.1))
    ((13.05 1.73 2.04 12.4 92 2.72 3.27 0.17 2.91 7.2 1.12 2.91 1150) (0.9 0.1 0.1))
    ((13.83 1.65 2.6 17.2 94 2.45 2.99 0.22 2.29 5.6 1.24 3.37 1265) (0.9 0.1 0.1))
    ((13.82 1.75 2.42 14 111 3.88 3.74 0.32 1.87 7.05 1.01 3.26 1190) (0.9 0.1 0.1))
    ((13.77 1.9 2.68 17.1 115 3 2.79 0.39 1.68 6.3 1.13 2.93 1375) (0.9 0.1 0.1))
    ((13.74 1.67 2.25 16.4 118 2.6 2.9 0.21 1.62 5.85 0.92 3.2 1060) (0.9 0.1 0.1))
    ((13.56 1.73 2.46 20.5 116 2.96 2.78 0.2 2.45 6.25 0.98 3.03 1120) (0.9 0.1 0.1))
    ((14.22 1.7 2.3 16.3 118 3.2 3 0.26 2.03 6.38 0.94 3.31 970) (0.9 0.1 0.1))
    ((13.29 1.97 2.68 16.8 102 3 3.23 0.31 1.66 6 1.07 2.84 1270) (0.9 0.1 0.1))
    ((13.72 1.43 2.5 16.7 108 3.4 3.67 0.19 2.04 6.8 0.89 2.87 1285) (0.9 0.1 0.1))
    ((12.37 0.94 1.36 10.6 88 1.98 0.57 0.28 0.42 1.95 1.05 1.82 520) (0.1 0.9 0.1))
    ((12.33 1.1 2.28 16 101 2.05 1.09 0.63 0.41 3.27 1.25 1.67 680) (0.1 0.9 0.1))
    ((12.64 1.36 2.02 16.8 100 2.02 1.41 0.53 0.62 5.75 0.98 1.59 450) (0.1 0.9 0.1))
    ((13.67 1.25 1.92 18 94 2.1 1.79 0.32 0.73 3.8 1.23 2.46 630) (0.1 0.9 0.1))
    ((12.37 1.13 2.16 19 87 3.5 3.1 0.19 1.87 4.45 1.22 2.87 420) (0.1 0.9 0.1))
    ((12.17 1.45 2.53 19 104 1.89 1.75 0.45 1.03 2.95 1.45 2.23 355) (0.1 0.9 0.1))
    ((12.37 1.21 2.56 18.1 98 2.42 2.65 0.37 2.08 4.6 1.19 2.3 678) (0.1 0.9 0.1))
    ((13.11 1.01 1.7 15 78 2.98 3.18 0.26 2.28 5.3 1.12 3.18 502) (0.1 0.9 0.1))
    ((12.37 1.17 1.92 19.6 78 2.11 2 0.27 1.04 4.68 1.12 3.48 510) (0.1 0.9 0.1))
    ((13.34 0.94 2.36 17 110 2.53 1.3 0.55 0.42 3.17 1.02 1.93 750) (0.1 0.9 0.1))
    ((12.21 1.19 1.75 16.8 151 1.85 1.28 0.14 2.5 2.85 1.28 3.07 718) (0.1 0.9 0.1))
    ((12.29 1.61 2.21 20.4 103 1.1 1.02 0.37 1.46 3.05 0.906 1.82 870) (0.1 0.9 0.1))
    ((13.86 1.51 2.67 25 86 2.95 2.86 0.21 1.87 3.38 1.36 3.16 410) (0.1 0.9 0.1))
    ((13.49 1.66 2.24 24 87 1.88 1.84 0.27 1.03 3.74 0.98 2.78 472) (0.1 0.9 0.1))
    ((12.99 1.67 2.6 30 139 3.3 2.89 0.21 1.96 3.35 1.31 3.5 985) (0.1 0.9 0.1))
    ((11.96 1.09 2.3 21 101 3.38 2.14 0.13 1.65 3.21 0.99 3.13 886) (0.1 0.9 0.1))
    ((11.66 1.88 1.92 16 97 1.61 1.57 0.34 1.15 3.8 1.23 2.14 428) (0.1 0.9 0.1))
    ((13.03 0.9 1.71 16 86 1.95 2.03 0.24 1.46 4.6 1.19 2.48 392) (0.1 0.9 0.1))
    ((11.84 2.89 2.23 18 112 1.72 1.32 0.43 0.95 2.65 0.96 2.52 500) (0.1 0.9 0.1))
    ((12.33 0.99 1.95 14.8 136 1.9 1.85 0.35 2.76 3.4 1.06 2.31 750) (0.1 0.9 0.1))
    ((12.7 3.87 2.4 23 101 2.83 2.55 0.43 1.95 2.57 1.19 3.13 463) (0.1 0.9 0.1))
    ((12 0.92 2 19 86 2.42 2.26 0.3 1.43 2.5 1.38 3.12 278) (0.1 0.9 0.1))
    ((12.72 1.81 2.2 18.8 86 2.2 2.53 0.26 1.77 3.9 1.16 3.14 714) (0.1 0.9 0.1))
    ((12.08 1.13 2.51 24 78 2 1.58 0.4 1.4 2.2 1.31 2.72 630) (0.1 0.9 0.1))
    ((13.05 3.86 2.32 22.5 85 1.65 1.59 0.61 1.62 4.8 0.84 2.01 515) (0.1 0.9 0.1))
    ((11.84 0.89 2.58 18 94 2.2 2.21 0.22 2.35 3.05 0.79 3.08 520) (0.1 0.9 0.1))
    ((12.67 0.98 2.24 18 99 2.2 1.94 0.3 1.46 2.62 1.23 3.16 450) (0.1 0.9 0.1))
    ((12.16 1.61 2.31 22.8 90 1.78 1.69 0.43 1.56 2.45 1.33 2.26 495) (0.1 0.9 0.1))
    ((11.65 1.67 2.62 26 88 1.92 1.61 0.4 1.34 2.6 1.36 3.21 562) (0.1 0.9 0.1))
    ((11.64 2.06 2.46 21.6 84 1.95 1.69 0.48 1.35 2.8 1 2.75 680) (0.1 0.9 0.1))
    ((12.08 1.33 2.3 23.6 70 2.2 1.59 0.42 1.38 1.74 1.07 3.21 625) (0.1 0.9 0.1))
    ((12.08 1.83 2.32 18.5 81 1.6 1.5 0.52 1.64 2.4 1.08 2.27 480) (0.1 0.9 0.1))
    ((12 1.51 2.42 22 86 1.45 1.25 0.5 1.63 3.6 1.05 2.65 450) (0.1 0.9 0.1))
    ((12.69 1.53 2.26 20.7 80 1.38 1.46 0.58 1.62 3.05 0.96 2.06 495) (0.1 0.9 0.1))
    ((12.29 2.83 2.22 18 88 2.45 2.25 0.25 1.99 2.15 1.15 3.3 290) (0.1 0.9 0.1))
    ((11.62 1.99 2.28 18 98 3.02 2.26 0.17 1.35 3.25 1.16 2.96 345) (0.1 0.9 0.1))
    ((12.47 1.52 2.2 19 162 2.5 2.27 0.32 3.28 2.6 1.16 2.63 937) (0.1 0.9 0.1))
    ((11.81 2.12 2.74 21.5 134 1.6 0.99 0.14 1.56 2.5 0.95 2.26 625) (0.1 0.9 0.1))
    ((12.29 1.41 1.98 16 85 2.55 2.5 0.29 1.77 2.9 1.23 2.74 428) (0.1 0.9 0.1))
    ((12.37 1.07 2.1 18.5 88 3.52 3.75 0.24 1.95 4.5 1.04 2.77 660) (0.1 0.9 0.1))
    ((12.29 3.17 2.21 18 88 2.85 2.99 0.45 2.81 2.3 1.42 2.83 406) (0.1 0.9 0.1))
    ((12.08 2.08 1.7 17.5 97 2.23 2.17 0.26 1.4 3.3 1.27 2.96 710) (0.1 0.9 0.1))
    ((12.6 1.34 1.9 18.5 88 1.45 1.36 0.29 1.35 2.45 1.04 2.77 562) (0.1 0.9 0.1))
    ((12.34 2.45 2.46 21 98 2.56 2.11 0.34 1.31 2.8 0.8 3.38 438) (0.1 0.9 0.1))
    ((11.82 1.72 1.88 19.5 86 2.5 1.64 0.37 1.42 2.06 0.94 2.44 415) (0.1 0.9 0.1))
    ((12.51 1.73 1.98 20.5 85 2.2 1.92 0.32 1.48 2.94 1.04 3.57 672) (0.1 0.9 0.1))
    ((12.42 2.55 2.27 22 90 1.68 1.84 0.66 1.42 2.7 0.86 3.3 315) (0.1 0.9 0.1))
    ((12.25 1.73 2.12 19 80 1.65 2.03 0.37 1.63 3.4 1 3.17 510) (0.1 0.9 0.1))
    ((12.72 1.75 2.28 22.5 84 1.38 1.76 0.48 1.63 3.3 0.88 2.42 488) (0.1 0.9 0.1))
    ((12.22 1.29 1.94 19 92 2.36 2.04 0.39 2.08 2.7 0.86 3.02 312) (0.1 0.9 0.1))
    ((11.61 1.35 2.7 20 94 2.74 2.92 0.29 2.49 2.65 0.96 3.26 680) (0.1 0.9 0.1))
    ((11.46 3.74 1.82 19.5 107 3.18 2.58 0.24 3.58 2.9 0.75 2.81 562) (0.1 0.9 0.1))
    ((12.52 2.43 2.17 21 88 2.55 2.27 0.26 1.22 2 0.9 2.78 325) (0.1 0.9 0.1))
    ((11.76 2.68 2.92 20 103 1.75 2.03 0.6 1.05 3.8 1.23 2.5 607) (0.1 0.9 0.1))
    ((11.41 0.74 2.5 21 88 2.48 2.01 0.42 1.44 3.08 1.1 2.31 434) (0.1 0.9 0.1))
    ((12.08 1.39 2.5 22.5 84 2.56 2.29 0.43 1.04 2.9 0.93 3.19 385) (0.1 0.9 0.1))
    ((11.03 1.51 2.2 21.5 85 2.46 2.17 0.52 2.01 1.9 1.71 2.87 407) (0.1 0.9 0.1))
    ((11.82 1.47 1.99 20.8 86 1.98 1.6 0.3 1.53 1.95 0.95 3.33 495) (0.1 0.9 0.1))
    ((12.42 1.61 2.19 22.5 108 2 2.09 0.34 1.61 2.06 1.06 2.96 345) (0.1 0.9 0.1))
    ((12.77 3.43 1.98 16 80 1.63 1.25 0.43 0.83 3.4 0.7 2.12 372) (0.1 0.9 0.1))
    ((12 3.43 2 19 87 2 1.64 0.37 1.87 1.28 0.93 3.05 564) (0.1 0.9 0.1))
    ((11.45 2.4 2.42 20 96 2.9 2.79 0.32 1.83 3.25 0.8 3.39 625) (0.1 0.9 0.1))
    ((11.56 2.05 3.23 28.5 119 3.18 5.08 0.47 1.87 6 0.93 3.69 465) (0.1 0.9 0.1))
    ((12.42 4.43 2.73 26.5 102 2.2 2.13 0.43 1.71 2.08 0.92 3.12 365) (0.1 0.9 0.1))
    ((13.05 5.8 2.13 21.5 86 2.62 2.65 0.3 2.01 2.6 0.73 3.1 380) (0.1 0.9 0.1))
    ((11.87 4.31 2.39 21 82 2.86 3.03 0.21 2.91 2.8 0.75 3.64 380) (0.1 0.9 0.1))
    ((12.07 2.16 2.17 21 85 2.6 2.65 0.37 1.35 2.76 0.86 3.28 378) (0.1 0.9 0.1))
    ((12.43 1.53 2.29 21.5 86 2.74 3.15 0.39 1.77 3.94 0.69 2.84 352) (0.1 0.9 0.1))
    ((11.79 2.13 2.78 28.5 92 2.13 2.24 0.58 1.76 3 0.97 2.44 466) (0.1 0.9 0.1))
    ((12.37 1.63 2.3 24.5 88 2.22 2.45 0.4 1.9 2.12 0.89 2.78 342) (0.1 0.9 0.1))
    ((12.04 4.3 2.38 22 80 2.1 1.75 0.42 1.35 2.6 0.79 2.57 580) (0.1 0.9 0.1))
    ((12.86 1.35 2.32 18 122 1.51 1.25 0.21 0.94 4.1 0.76 1.29 630) (0.1 0.1 0.9))
    ((12.88 2.99 2.4 20 104 1.3 1.22 0.24 0.83 5.4 0.74 1.42 530) (0.1 0.1 0.9))
    ((12.81 2.31 2.4 24 98 1.15 1.09 0.27 0.83 5.7 0.66 1.36 560) (0.1 0.1 0.9))
    ((12.7 3.55 2.36 21.5 106 1.7 1.2 0.17 0.84 5 0.78 1.29 600) (0.1 0.1 0.9))
    ((12.51 1.24 2.25 17.5 85 2 0.58 0.6 1.25 5.45 0.75 1.51 650) (0.1 0.1 0.9))
    ((12.6 2.46 2.2 18.5 94 1.62 0.66 0.63 0.94 7.1 0.73 1.58 695) (0.1 0.1 0.9))
    ((12.25 4.72 2.54 21 89 1.38 0.47 0.53 0.8 3.85 0.75 1.27 720) (0.1 0.1 0.9))
    ((12.53 5.51 2.64 25 96 1.79 0.6 0.63 1.1 5 0.82 1.69 515) (0.1 0.1 0.9))
    ((13.49 3.59 2.19 19.5 88 1.62 0.48 0.58 0.88 5.7 0.81 1.82 580) (0.1 0.1 0.9))
    ((12.84 2.96 2.61 24 101 2.32 0.6 0.53 0.81 4.92 0.89 2.15 590) (0.1 0.1 0.9))
    ((12.93 2.81 2.7 21 96 1.54 0.5 0.53 0.75 4.6 0.77 2.31 600) (0.1 0.1 0.9))
    ((13.36 2.56 2.35 20 89 1.4 0.5 0.37 0.64 5.6 0.7 2.47 780) (0.1 0.1 0.9))
    ((13.52 3.17 2.72 23.5 97 1.55 0.52 0.5 0.55 4.35 0.89 2.06 520) (0.1 0.1 0.9))
    ((13.62 4.95 2.35 20 92 2 0.8 0.47 1.02 4.4 0.91 2.05 550) (0.1 0.1 0.9))
    ((12.25 3.88 2.2 18.5 112 1.38 0.78 0.29 1.14 8.21 0.65 2 855) (0.1 0.1 0.9))
    ((13.16 3.57 2.15 21 102 1.5 0.55 0.43 1.3 4 0.6 1.68 830) (0.1 0.1 0.9))
    ((13.88 5.04 2.23 20 80 0.98 0.34 0.4 0.68 4.9 0.58 1.33 415) (0.1 0.1 0.9))
    ((12.87 4.61 2.48 21.5 86 1.7 0.65 0.47 0.86 7.65 0.54 1.86 625) (0.1 0.1 0.9))
    ((13.32 3.24 2.38 21.5 92 1.93 0.76 0.45 1.25 8.42 0.55 1.62 650) (0.1 0.1 0.9))
    ((13.08 3.9 2.36 21.5 113 1.41 1.39 0.34 1.14 9.4 0.57 1.33 550) (0.1 0.1 0.9))
    ((13.5 3.12 2.62 24 123 1.4 1.57 0.22 1.25 8.6 0.59 1.3 500) (0.1 0.1 0.9))
    ((12.79 2.67 2.48 22 112 1.48 1.36 0.24 1.26 10.8 0.48 1.47 480) (0.1 0.1 0.9))
    ((13.11 1.9 2.75 25.5 116 2.2 1.28 0.26 1.56 7.1 0.61 1.33 425) (0.1 0.1 0.9))
    ((13.23 3.3 2.28 18.5 98 1.8 0.83 0.61 1.87 10.52 0.56 1.51 675) (0.1 0.1 0.9))
    ((12.58 1.29 2.1 20 103 1.48 0.58 0.53 1.4 7.6 0.58 1.55 640) (0.1 0.1 0.9))
    ((13.17 5.19 2.32 22 93 1.74 0.63 0.61 1.55 7.9 0.6 1.48 725) (0.1 0.1 0.9))
    ((13.84 4.12 2.38 19.5 89 1.8 0.83 0.48 1.56 9.01 0.57 1.64 480) (0.1 0.1 0.9))
    ((12.45 3.03 2.64 27 97 1.9 0.58 0.63 1.14 7.5 0.67 1.73 880) (0.1 0.1 0.9))
    ((14.34 1.68 2.7 25 98 2.8 1.31 0.53 2.7 13 0.57 1.96 660) (0.1 0.1 0.9))
    ((13.48 1.67 2.64 22.5 89 2.6 1.1 0.52 2.29 11.75 0.57 1.78 620) (0.1 0.1 0.9))
    ((12.36 3.83 2.38 21 88 2.3 0.92 0.5 1.04 7.65 0.56 1.58 520) (0.1 0.1 0.9))
    ((13.69 3.26 2.54 20 107 1.83 0.56 0.5 0.8 5.88 0.96 1.82 680) (0.1 0.1 0.9))
    ((12.85 3.27 2.58 22 106 1.65 0.6 0.6 0.96 5.58 0.87 2.11 570) (0.1 0.1 0.9))
    ((12.96 3.45 2.35 18.5 106 1.39 0.7 0.4 0.94 5.28 0.68 1.75 675) (0.1 0.1 0.9))
    ((13.78 2.76 2.3 22 90 1.35 0.68 0.41 1.03 9.58 0.7 1.68 615) (0.1 0.1 0.9))
    ((13.73 4.36 2.26 22.5 88 1.28 0.47 0.52 1.15 6.62 0.78 1.75 520) (0.1 0.1 0.9))
    ((13.45 3.7 2.6 23 111 1.7 0.92 0.43 1.46 10.68 0.85 1.56 695) (0.1 0.1 0.9))
    ((12.82 3.37 2.3 19.5 88 1.48 0.66 0.4 0.97 10.26 0.72 1.75 685) (0.1 0.1 0.9))
    ((13.58 2.58 2.69 24.5 105 1.55 0.84 0.39 1.54 8.66 0.74 1.8 750) (0.1 0.1 0.9))
    ((13.4 4.6 2.86 25 112 1.98 0.96 0.27 1.11 8.5 0.67 1.92 630) (0.1 0.1 0.9))
    ((12.2 3.03 2.32 19 96 1.25 0.49 0.4 0.73 5.5 0.66 1.83 510) (0.1 0.1 0.9))
    ((12.77 2.39 2.28 19.5 86 1.39 0.51 0.48 0.64 9.899999 0.57 1.63 470) (0.1 0.1 0.9))
    ((14.16 2.51 2.48 20 91 1.68 0.7 0.44 1.24 9.7 0.62 1.71 660) (0.1 0.1 0.9))
    ((13.71 5.65 2.45 20.5 95 1.68 0.61 0.52 1.06 7.7 0.64 1.74 740) (0.1 0.1 0.9))
    ((13.4 3.91 2.48 23 102 1.8 0.75 0.43 1.41 7.3 0.7 1.56 750) (0.1 0.1 0.9))
    ((13.27 4.28 2.26 20 120 1.59 0.69 0.43 1.35 10.2 0.59 1.56 835) (0.1 0.1 0.9))
    ((13.17 2.59 2.37 20 120 1.65 0.68 0.53 1.46 9.3 0.6 1.62 840) (0.1 0.1 0.9))
    ((14.13 4.1 2.74 24.5 96 2.05 0.76 0.56 1.35 9.2 0.61 1.6 560) (0.1 0.1 0.9))
    ))


(defun test-cases ()
	(let ((temp *debug*))
		;;i dont want debug messages on for printing test cases
		(setf *debug* '())
		
		(print "NET-ERRROR: output should be (-3 -1 1 3)")
		(print (net-error '(1 2 3 4) '(4 3 2 1)))
		(print "NET-BUILD: output should be: ((4x3 matrix) (4*1 matrix)) with values between -9 and 9. this represents 4 hidden nodes, 3 inputs, and 1 output")
		(print (net-build (convert-datum *xor*) 4 .2 9 90 2))
		(print "FORWARD-PROPAGATE: should return values from each layer so ((input vector) (some-hidden-layer-vector) (answer-vector))")
		
		;;(first (first data)) gets the first set of input. (first (second data)) would get the first output set
		(print (forward-propagate (first (first (convert-datum *xor*))) (net-build (convert-datum *xor*) 3 .2 9 90 2)))
		
		(print "full data training test")
		(print (full-data-training (convert-datum *xor*) 4 .2 1 1))
		;;set the debug state to whatever it was before i set it to nil
		(setf *debug* temp)))


#|
;;; some test code for you to try

;;; you'll need to run this a couple of times before it globally converges.
;;; When it *doesn't* converge what is usually happening?
(net-build (convert-datum *nand*) 3 1.0 5 20000 1000 t)

(net-build (convert-datum *xor*) 3 1.0 5 20000 1000 t)


;; how well does this converge on average?  Can you modify it to do better?
(net-build (convert-datum *voting-records*) 10 1.0 2 5000 250 t)


;;; how well does this generalize usually?  Can you modify it to typically generalize better?
(simple-generalization *voting-records* ...)  ;; pick appropriate values

;;; how well does this generalize usually?  Can you modify it to typically generalize better?
(simple-generalization *mpg* ...) ;; pick appropriate values

;;; how well does this generalize usually?  Can you modify it to typically generalize better?
(simple-generalization *wine* ...)  ;; pick appropriate values

|#


;;test cases for each function

;;main?
(setf *debug* t)
(test-cases)
(print (full-data-training (convert-datum *xor*) 4 .2 1 1))
(print (net-build (convert-datum *xor*) 4 .2 9 90 2))
(print (forward-propagate (dprint (first (first (convert-datum *xor*))) "CONVERTED-DATA") (net-build (convert-datum *xor*) 3 .2 9 90 2)))

(print "HELLO: here's a full output")
(print (full-data-training (convert-datum *xor*) 4 .2 1 1))
rowSS <- function(x, cols) {
    apply(x, 1, function(v) {
        v <- v[cols]
        sum((v - mean(v))^2)
    })
}

#' @export
refVariancePartition <- function(x, centers) {
    assignments <- rep(1L, ncol(x))
    rowvars <- list(rowSS(x, seq_len(ncol(x))))
    variances <- sum(rowvars[[1]])

    for (steps in seq_len(centers - 1)) {
        if (length(variances) != 0) {
            maxed <- which.max(variances) 
        } else {
            maxed <- 1L
        }

        chosen <- assignments == maxed
        rv <- rowvars[[maxed]]
        d <- which.max(rv)

        coords <- x[d,chosen] 
        newer <- mean(coords) < coords
        count <- steps + 1L
        assignments[chosen][newer] <- count

        oldies <- which(assignments == maxed)
        rowvars[[maxed]] <- rowSS(x, cols=oldies)
        variances[maxed] <- sum(rowvars[[maxed]])

        newbs <- which(assignments == count)
        rowvars[[count]] <- rowSS(x, cols=newbs)
        variances[count] <- sum(rowvars[[count]])
    }

    mat <- matrix(0, nrow(x), centers)
    for (i in seq_len(centers)) {
        mat[,i] <- rowMeans(x[,assignments == i,drop=FALSE])
    }
    mat
}

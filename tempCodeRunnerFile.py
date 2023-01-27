x = len(pd.DataFrame(dict(uid = lp1, isbn = lp2)))
            subDfs = 16
            for j in range(0, x, x // subDfs):
                tmpDf = usersPerCluster['uid'].iloc[j: j + x // subDfs]
                lp1, lp2 = pd.core.reshape.util.cartesian_product([tmpDf.to_list(), ratingsPerCluster['isbn'].to_list()])
                finalDf = pd.DataFrame(dict(uid = lp1, isbn = lp2))
                finalDf['uid'] = finalDf['uid'].astype(str)
                finalDf = pd.merge(finalDf, allRatingsDf, how = 'left', on = ['uid', 'isbn'])
                finalDf = pd.merge(finalDf, ratingsPerCluster, how = 'left', on = ['isbn'])
                finalDf.rating.fillna(finalDf.av_rating, inplace=True)
                del finalDf['av_rating']
                del finalDf['cluster']
                print(finalDf)
                print(i, ': iteration', j, ': subDf')
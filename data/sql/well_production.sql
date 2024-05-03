WITH wells AS (
	SELECT wm.WellNumber AS well_name,
		wm.api_wellno,
		p.PoolName
	FROM [tblWellFieldPool] w
	INNER JOIN [tblRefGeoPool] p
	ON w.FieldPoolCode = p.FieldPoolCode
	LEFT JOIN [tblWellMaster] wm 
	ON w.API_WellNo = wm.API_WellNo
	WHERE p.PoolName = 'KUPARUK RIVER, KUPARUK RIV OIL'
),
prod AS (
	SELECT w.well_name,
		w.PoolName,
		p.reportdate,
		p.prodoil,
		p.prodwater,
		p.prodgas,
		p.proddays
	FROM [tblWellPoolProduction] p
	INNER JOIN wells w
	ON w.api_wellno = p.api_wellno
)
SELECT *
FROM prod
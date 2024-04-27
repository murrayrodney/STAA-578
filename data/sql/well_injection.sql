WITH wells AS (
	SELECT wm.WellNumber AS well_name,
		wm.api_wellno
	FROM [tblWellFieldPool] w
	INNER JOIN [tblRefGeoPool] p
	ON w.FieldPoolCode = p.FieldPoolCode
	LEFT JOIN [tblWellMaster] wm 
	ON w.API_WellNo = wm.API_WellNo
	WHERE p.PoolName = 'PRUDHOE BAY, PRUDHOE OIL'
	AND wm.bh_ns IS NOT null 
),
inj AS (
	SELECT w.well_name,
		i.reportdate,
		i.welltype,
		i.vol_liq,
		i.vol_gas
	FROM [tblWellPoolInjection] i
	INNER JOIN wells w
	ON w.api_wellno = i.api_wellno
	WHERE welltype = 4
)
SELECT *
FROM inj
